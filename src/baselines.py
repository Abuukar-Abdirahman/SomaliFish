"""Baselines the hotspot model must beat before it can claim any value.

docs/ARCHITECTURE.md makes this non-negotiable, and rightly: a model that
cannot beat "look at the depth" or "look at what happened last year" has not
learned anything about the ocean, however good its ROC curve looks.

Two baselines:

  depth_only    -- gradient-boosted trees on bathymetry alone. Fishing is
                   strongly depth-structured (shelf edges, banks), so this is
                   a genuinely hard bar, not a straw man.
  climatology   -- for each cell and calendar month, the rate at which fishing
                   occurred there in the training years. No ocean data at all,
                   just "this is where boats usually go in August". Beating
                   this is what proves the model responds to conditions rather
                   than memorising a map.

Also here: navigable_days(), the filter that addresses the monsoon confound.

Usage:
    python src/baselines.py                # run both on whatever years exist
    python src/baselines.py --train-end 2023
"""

from __future__ import annotations

import argparse
import glob
import sys

import numpy as np
import pandas as pd

import config


# ------------------------------------------------------------- the confound

def navigable_days(df: pd.DataFrame,
                   wave_max_m: float | None = None,
                   wind_max_kmh: float | None = None) -> pd.Series:
    """Rows where the sea was workable enough that a vessel had a real choice.

    Why this matters. Apparent fishing effort collapses in July-August, when
    the SW monsoon makes the sea dangerous -- and that is exactly when Somali
    upwelling makes the water most productive. In the raw labels, chlorophyll
    is NEGATIVELY correlated with fishing. A model trained on all days learns
    "rich green water means no fish", which is backwards, and would tell
    fishermen to avoid the sea in its most productive season.

    Restricting training to navigable days changes the question from
        "did anyone fish here?"           (mostly a weather question)
    to
        "given that boats were working, did they choose HERE?"
    which is the question we actually want answered.

    Defaults to the CAUTION threshold: conditions a working vessel would
    accept, not merely flat calm.
    """
    wave_cap = config.WAVE_DANGER_M if wave_max_m is None else wave_max_m
    wind_cap = config.WIND_DANGER_KMH if wind_max_kmh is None else wind_max_kmh

    wave = df.get("wave_height_max")
    wind = df.get("wind_speed_max")
    if wave is None or wind is None:
        print("[warn] no wave/wind columns - cannot filter to navigable days")
        return pd.Series(True, index=df.index)

    ok = (wave <= wave_cap) & (wind <= wind_cap)
    # Unknown conditions are excluded: we cannot claim a vessel had a choice.
    return ok.fillna(False)


def fleet_active_months(df: pd.DataFrame, min_positive_rate: float = 0.002
                        ) -> tuple[pd.Series, list[int]]:
    """Rows in months when the AIS fleet was actually present in the region.

    This, not navigable_days(), is the real fix for the monsoon confound.

    Measured on 2019: in July the whole Somali EEZ saw 62 vessel-days of
    apparent fishing, against 5,316 in January -- a 99% collapse -- while
    36,000 navigable cell-days still existed. The distant-water fleets
    (Chinese and Iranian, 85% of effort) do not wait out the SW monsoon in
    port; they leave the western Indian Ocean altogether. So "no fishing in
    August" carries no information about whether August water holds fish, and
    no per-day weather filter can recover it: the vessels were not there to
    choose.

    Training across those months teaches the model that the richest water of
    the year is the worst, which is backwards. We therefore train only where
    the fleet was present, and state plainly that the hotspot model does not
    apply in the off-season. The safety module and the physics index still do.
    """
    if "fished" not in df:
        return pd.Series(True, index=df.index), []
    rate = df.groupby("month", observed=True)["fished"].mean()

    # Detection relies on months differing in prevalence. If negatives were
    # subsampled to a fixed ratio per period (as the wide region is, to fit in
    # memory), every month ends up at the same rate by construction and
    # nothing looks dormant. Fall back to the months measured directly from
    # unsubsampled Somali data rather than silently keeping everything.
    spread = float(rate.max() - rate.min()) if len(rate) else 0.0
    if spread < 0.2 * float(rate.mean() or 1):
        active = sorted(set(range(1, 13)) - set(config.FLEET_DORMANT_MONTHS))
        print(f"  [note] monthly positive rates are near-uniform "
              f"(spread {spread:.4f}) - this data looks class-balanced, so "
              f"dormancy cannot be detected from it.\n"
              f"         Falling back to the measured dormant months "
              f"{config.FLEET_DORMANT_MONTHS}.")
        return df["month"].isin(active), active

    active = sorted(int(m) for m in rate[rate >= min_positive_rate].index)
    dormant = sorted(set(range(1, 13)) - set(active))
    if dormant:
        print(f"  fleet-dormant months excluded: {dormant} "
              f"(positive rate < {min_positive_rate:.3%})")
    return df["month"].isin(active), active


def evaluate_within_month(df: pd.DataFrame, y_score, label: str = "fished",
                          name: str = "model") -> dict:
    """PR-AUC computed per month, then averaged over months.

    A model can score well overall just by knowing "April is busy, August is
    not" -- seasonal level, no spatial skill. Scoring inside each month strips
    that out and asks the question that matters to a fisherman: given today,
    WHERE should the boat go?
    """
    from sklearn.metrics import average_precision_score

    work = df[[label, "month"]].copy()
    work["score"] = np.asarray(y_score, dtype="float64")
    work = work[np.isfinite(work["score"])]

    rows = []
    for month, grp in work.groupby("month", observed=True):
        if grp[label].nunique() < 2:
            continue
        base = float(grp[label].mean())
        pr = float(average_precision_score(grp[label], grp["score"]))
        rows.append({"month": int(month), "n": len(grp), "base": base,
                     "pr_auc": pr, "lift": pr / base if base else np.nan})
    if not rows:
        return {"model": name, "note": "no month had both classes"}

    per_month = pd.DataFrame(rows)
    return {
        "model": name,
        "months": len(per_month),
        "pr_auc_within_month": round(float(per_month["pr_auc"].mean()), 4),
        "lift_within_month": round(float(per_month["lift"].mean()), 2),
        "per_month": per_month.round(4).to_dict("records"),
    }


# ------------------------------------------------------------------ metrics

def evaluate(y_true, y_score, name: str) -> dict:
    """ROC-AUC, PR-AUC and lift. PR-AUC is the one that matters here.

    Positives are ~1% of rows, so accuracy is meaningless (predict "no
    fishing" everywhere and score 99%). PR-AUC against the base rate is the
    honest measure of whether the ranking is useful.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype="float64")
    ok = np.isfinite(y_score)
    y_true, y_score = y_true[ok], y_score[ok]

    base = float(y_true.mean()) if len(y_true) else float("nan")
    out = {"model": name, "n": int(len(y_true)), "base_rate": round(base, 5)}
    if len(np.unique(y_true)) < 2:
        out["note"] = "only one class present - metrics undefined"
        return out

    out["roc_auc"] = round(float(roc_auc_score(y_true, y_score)), 4)
    pr = float(average_precision_score(y_true, y_score))
    out["pr_auc"] = round(pr, 4)
    # How much better than guessing at the base rate.
    out["pr_lift"] = round(pr / base, 2) if base > 0 else None
    return out


# ---------------------------------------------------------------- baselines

def physics_index_baseline(train: pd.DataFrame, test: pd.DataFrame,
                           label: str = "fished") -> np.ndarray:
    """Score the physics index the map actually ships, on held-out data.

    Not a baseline to beat -- it is the thing currently being served, so it
    needs testing at least as much as the model does. It is unsupervised: it
    ignores `train` entirely and just ranks measured chlorophyll and front
    strength. That means it cannot overfit, but it also has no guarantee of
    being right, and until this is run there is no evidence either way.

    Ranked per day, matching how the map presents it (a cell is promising
    relative to the rest of the grid that day, not on an absolute scale).
    """
    import zones as zonelib

    out = np.full(len(test), np.nan, dtype="float64")
    positions = {d: np.flatnonzero((test["date"] == d).to_numpy())
                 for d in test["date"].unique()}
    for day, idx in positions.items():
        if len(idx) == 0:
            continue
        out[idx] = zonelib.physics_index(test.iloc[idx]).to_numpy(dtype="float64")
    return out


def depth_only(train: pd.DataFrame, test: pd.DataFrame,
               label: str = "fished") -> np.ndarray:
    """Predict fishing from bathymetry alone."""
    import xgboost as xgb

    feats = [c for c in ("depth", "seabed_roughness") if c in train.columns]
    if not feats:
        raise SystemExit("no depth column - run build_dataset.py with GEBCO present")

    dtrain = xgb.DMatrix(train[feats].astype("float32"), label=train[label])
    dtest = xgb.DMatrix(test[feats].astype("float32"))
    params = {"objective": "binary:logistic", "eval_metric": "aucpr",
              "max_depth": 4, "eta": 0.1, "subsample": 0.8,
              "colsample_bytree": 1.0, "nthread": 4, "seed": 42}
    booster = xgb.train(params, dtrain, num_boost_round=120)
    return booster.predict(dtest)


def climatology(train: pd.DataFrame, test: pd.DataFrame,
                label: str = "fished") -> np.ndarray:
    """Predict each cell-month's historical fishing rate from the train years.

    Uses no ocean conditions whatsoever. Backs off cell-month -> cell -> month
    -> global whenever a combination was never observed in training.
    """
    key = ["cell_id", "month"]
    cell_month = train.groupby(key, observed=True)[label].mean()
    by_cell = train.groupby("cell_id", observed=True)[label].mean()
    by_month = train.groupby("month", observed=True)[label].mean()
    overall = float(train[label].mean())

    idx = pd.MultiIndex.from_arrays([test["cell_id"], test["month"]])
    # .copy(): reindex can hand back a read-only view, and we fill gaps below.
    pred = cell_month.reindex(idx).to_numpy(dtype="float64").copy()

    gaps = np.isnan(pred)
    if gaps.any():
        pred[gaps] = by_cell.reindex(test["cell_id"][gaps]).to_numpy(dtype="float64")
    gaps = np.isnan(pred)
    if gaps.any():
        pred[gaps] = by_month.reindex(test["month"][gaps]).to_numpy(dtype="float64")
    pred[np.isnan(pred)] = overall
    return pred


# -------------------------------------------------------------------- data

def load_years(years: list[int] | None = None,
               columns: list[str] | None = None) -> pd.DataFrame:
    """Load processed historical feature tables."""
    files = sorted(glob.glob(str(config.PROCESSED_DIR / "features_historical_*.parquet")))
    if years:
        files = [f for f in files if any(str(y) in f for y in years)]
    if not files:
        sys.exit("No processed historical files. Run: python src/build_dataset.py")
    print(f"Loading {len(files)} year file(s)...")
    frames = [pd.read_parquet(f, columns=columns) for f in files]
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year.astype("int16")
    return df


def temporal_split(df: pd.DataFrame, train_end_year: int | None = None):
    """Train on earlier years, test on later ones. Never a random split.

    Ocean fields are strongly autocorrelated in space and time; a random split
    puts near-duplicate rows on both sides and produces meaninglessly
    optimistic scores.
    """
    end = config.TRAIN_END_YEAR if train_end_year is None else train_end_year
    train = df[df["year"] <= end]
    test = df[df["year"] > end]
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-end", type=int, default=config.TRAIN_END_YEAR)
    parser.add_argument("--navigable-only", action="store_true",
                        help="restrict to navigable days (weather-choice filter)")
    parser.add_argument("--all-months", action="store_true",
                        help="do NOT drop fleet-dormant months (not advised: "
                             "the monsoon confound inverts the chlorophyll signal)")
    args = parser.parse_args()

    cols = ["cell_id", "date", "lat", "lon", "fished", "depth",
            "seabed_roughness", "month", "wave_height_max", "wind_speed_max"]
    df = load_years(columns=cols)
    print(f"{len(df):,} rows, years {df['year'].min()}-{df['year'].max()}")

    if not args.all_months:
        keep, active = fleet_active_months(df)
        print(f"fleet-active months {active}: keeping {int(keep.sum()):,} of "
              f"{len(df):,} rows ({keep.mean()*100:.1f}%)")
        df = df[keep]

    if args.navigable_only:
        keep = navigable_days(df)
        print(f"navigable-day filter: keeping {int(keep.sum()):,} of {len(df):,} "
              f"rows ({keep.mean()*100:.1f}%)")
        df = df[keep]

    train, test = temporal_split(df, args.train_end)
    print(f"train <= {args.train_end}: {len(train):,} rows "
          f"({train['fished'].mean()*100:.2f}% positive)")
    print(f"test  >  {args.train_end}: {len(test):,} rows "
          f"({test['fished'].mean()*100:.2f}% positive)")
    if test.empty or train.empty:
        sys.exit("\nNeed data on both sides of the split. Build more years first.")

    results, within = [], []
    for name, fn in (("depth_only", depth_only), ("climatology", climatology)):
        print(f"\nrunning {name}...")
        pred = fn(train, test)
        results.append(evaluate(test["fished"], pred, name))
        w = evaluate_within_month(test, pred, name=name)
        within.append({k: v for k, v in w.items() if k != "per_month"})

    print("\nOverall (pooled across months):")
    print(pd.DataFrame(results).to_string(index=False))
    print("\nWithin-month (strips out seasonal level; this is the honest test):")
    print(pd.DataFrame(within).to_string(index=False))
    print("\nThe hotspot model must beat BOTH baselines on within-month PR-AUC "
          "to claim it has learned anything about ocean conditions.")


if __name__ == "__main__":
    main()
