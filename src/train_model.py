"""Train and honestly validate the hotspot model.

STEP 3 of the pipeline. Trains an XGBoost classifier on "did AIS-tracked
vessels fish in this cell on this day, given the ocean conditions", calibrates
it, and refuses to bless it unless it beats both baselines.

Design decisions, and why:

  WEATHER IS EXCLUDED FROM THE FEATURES. Wave height and wind predict fishing
  very well - because boats avoid rough seas - but that is a weather forecast,
  not a fish forecast, and safety is already handled by a separate transparent
  rule module. Including them would let the model score well while learning
  nothing about fish. Override with --with-weather to see the difference.

  FLEET-DORMANT MONTHS ARE EXCLUDED. The distant-water fleet leaves the region
  during the SW monsoon, so July-August absence says nothing about fish. See
  baselines.fleet_active_months for the measurements.

  VALIDATION IS TEMPORAL, NEVER RANDOM. Train on earlier years, test on later
  ones. Ocean fields are heavily autocorrelated; a random split puts
  near-duplicate rows on both sides and inflates every metric.

  THE HEADLINE METRIC IS WITHIN-MONTH PR-AUC. Positives are ~1% of rows, so
  accuracy is meaningless, and pooled PR-AUC rewards merely knowing which
  months are busy. Within-month asks the useful question: given today, where?

Usage:
    python src/train_model.py                      # train, validate, save
    python src/train_model.py --spatial-holdout    # also test spatial transfer
    python src/train_model.py --dry-run            # report data readiness only
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import sys

import numpy as np
import pandas as pd

import baselines
import config

# Ocean-condition features. Deliberately no wave/wind - see module docstring.
FEATURES = [
    "sst", "sst_gradient",
    "chl", "chl_lag7", "chl_lag14", "chl_age_days",
    "current_speed",
    "depth", "seabed_roughness", "dist_coast_km",
    "month_sin", "month_cos",
    "lat", "lon",
]
WEATHER_FEATURES = ["wave_height_max", "wind_speed_max"]
LABEL = "fished"

MODEL_PATH = config.MODELS_DIR / "hotspot_xgb.json"
META_PATH = config.MODELS_DIR / "hotspot_xgb.meta.json"

# Kismayo lat band, the spatial holdout from docs/ARCHITECTURE.md.
SPATIAL_HOLDOUT_LAT = (-2.0, 1.5)

# Features available in BOTH regions. The wide training region has no GEBCO
# coverage (bathymetry is a manual per-box download), so depth, seabed
# roughness and distance-to-coast cannot be used in a transfer experiment.
# lat/lon are excluded too: the whole question is whether ocean conditions
# generalise to unseen water, and leaving coordinates in would let the model
# answer with geography instead.
TRANSFER_FEATURES = [
    "sst", "sst_gradient",
    "chl", "chl_lag7", "chl_lag14", "chl_age_days",
    "current_speed",
    "month_sin", "month_cos",
]


# ---------------------------------------------------------------- data load

def available_years(region: str = "somali") -> list[int]:
    prefix = ("features_historical_" if region == "somali"
              else f"features_{region}_historical_")
    files = glob.glob(str(config.PROCESSED_DIR / f"{prefix}*.parquet"))
    years = []
    for f in files:
        stem = f.rsplit("_", 1)[-1].split(".")[0]
        if stem.isdigit():
            years.append(int(stem))
    return sorted(years)


def region_file(region: str, year: int):
    prefix = ("features_historical_" if region == "somali"
              else f"features_{region}_historical_")
    return config.PROCESSED_DIR / f"{prefix}{year}.parquet"


def load_region(region: str, years: list[int], features: list[str],
                extra: list[str]) -> pd.DataFrame:
    """Load one region's processed years with only the columns needed."""
    want = sorted(set(features + extra))
    frames = []
    for year in years:
        path = region_file(region, year)
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=want)
        except Exception:
            df = pd.read_parquet(path)
            df = df[[c for c in want if c in df.columns]]
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year.astype("int16")
        frames.append(df)
        print(f"  {region} {year}: {len(df):,} rows, "
              f"{int(df[LABEL].sum()):,} positive")
    if not frames:
        sys.exit(f"No processed files for region '{region}'. "
                 f"Build them: python src/build_dataset.py --region {region}")
    return pd.concat(frames, ignore_index=True)


def run_transfer(args) -> None:
    """Train on the wider Indian Ocean, test on Somali waters.

    The question: does a model that has seen 7.8x more fishing across far more
    varied water learn an environment -> fishing relationship that carries
    into Somali waters it has never seen?

    Somali-box rows are removed from the training set, so this is a genuine
    spatial transfer test and not a memorisation check. A depth-free
    Somali-trained control is fitted on the same features, so the comparison
    is like for like.
    """
    features = list(TRANSFER_FEATURES)
    extra = [LABEL, "cell_id", "date", "month", "lat", "lon"]

    train_years = [y for y in available_years("wide") if y <= args.train_end]
    test_years = [y for y in available_years("somali") if y > args.train_end]
    print(f"Transfer experiment\n  train: wide region {train_years}"
          f"\n  test:  somali region {test_years}")
    if not train_years or not test_years:
        sys.exit("Need wide training years and Somali test years.")

    print("\nLoading wide training data...")
    wide = load_region("wide", train_years, features, extra)
    print("Loading Somali test data...")
    test = load_region("somali", test_years, features, extra)

    # Exclude the Somali box from training - otherwise this measures memory,
    # not transfer.
    box = config.REGIONS["somali"]
    inside = (wide["lat"].between(*box["lat"]) & wide["lon"].between(*box["lon"]))
    print(f"\nremoving {int(inside.sum()):,} Somali-box rows from wide training "
          f"({inside.mean()*100:.1f}%)")
    wide = wide[~inside]

    # Do NOT infer dormancy here. The wide files were class-balanced at build
    # time to fit in memory, which flattens every month to the same positive
    # rate - prevalence-based detection then either sees uniformity or, worse,
    # sees every month clearing the threshold and excludes nothing. Use the
    # months measured from unsubsampled Somali data instead.
    active = sorted(set(range(1, 13)) - set(config.FLEET_DORMANT_MONTHS))
    print(f"excluding fleet-dormant months {config.FLEET_DORMANT_MONTHS} "
          f"(measured on unsubsampled data, not inferred from these files)")
    wide = wide[wide["month"].isin(active)]
    test = test[test["month"].isin(active)]

    # XGBoost needs the whole matrix in memory at once; 10.5M x 9 float32 does
    # not fit here. Thin the training set, keeping every positive.
    if args.max_train_rows and len(wide) > args.max_train_rows:
        rng = np.random.default_rng(42)
        pos = wide[wide[LABEL] == 1]
        neg = wide[wide[LABEL] == 0]
        room = max(args.max_train_rows - len(pos), len(pos))
        if room < len(neg):
            neg = neg.iloc[np.sort(rng.choice(len(neg), size=room, replace=False))]
        wide = pd.concat([pos, neg], ignore_index=True)
        print(f"  capped training set at {len(wide):,} rows "
              f"(all {len(pos):,} positives kept)")

    print(f"\ntrain {len(wide):,} rows ({wide[LABEL].mean()*100:.2f}% positive)")
    print(f"test  {len(test):,} rows ({test[LABEL].mean()*100:.2f}% positive)")

    print("\nFitting wide-region model...")
    booster, calib = fit(wide, features)
    pred_wide = predict(booster, calib, test, features)

    # Control: same features, same split, but trained on Somali data only.
    print("\nFitting Somali-only control (same features)...")
    som_train_years = [y for y in available_years("somali") if y <= args.train_end]
    som = load_region("somali", som_train_years, features, extra)
    som = som[som["month"].isin(active)]     # same month filter as the wide side
    if args.neg_per_pos:
        rng = np.random.default_rng(42)
        pos, neg = som[som[LABEL] == 1], som[som[LABEL] == 0]
        take = min(len(neg), len(pos) * args.neg_per_pos)
        neg = neg.iloc[np.sort(rng.choice(len(neg), size=take, replace=False))]
        som = pd.concat([pos, neg], ignore_index=True)
    ctrl, ctrl_cal = fit(som, features)
    pred_ctrl = predict(ctrl, ctrl_cal, test, features)

    print("\nScoring baselines...")
    scored = [
        ("wide_transfer", pred_wide),
        ("somali_control", pred_ctrl),
        ("climatology", baselines.climatology(som, test)),
        ("physics_index", baselines.physics_index_baseline(som, test)),
    ]
    within = [baselines.evaluate_within_month(test, p, name=n) for n, p in scored]
    pooled = [baselines.evaluate(test[LABEL], p, n) for n, p in scored]

    print("\nPooled:")
    print(pd.DataFrame(pooled).to_string(index=False))
    print("\nWithin-month (the honest test):")
    print(pd.DataFrame([{k: v for k, v in w.items() if k != "per_month"}
                        for w in within]).to_string(index=False))

    wide_score = within[0]["pr_auc_within_month"]
    clim_score = next(w["pr_auc_within_month"] for w in within
                      if w["model"] == "climatology")
    ctrl_score = within[1]["pr_auc_within_month"]
    print(f"\nWide transfer beats climatology: {wide_score > clim_score}")
    print(f"Wide transfer beats Somali-only control: {wide_score > ctrl_score}")
    print("\nFeature importance (wide model):")
    for row in feature_importance(booster, features)[:8]:
        print(f"  {row['feature']:18s} {row['gain_share']*100:5.1f}%")

    out = config.MODELS_DIR / "transfer_experiment.json"
    out.write_text(json.dumps({
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "features": features,
        "train_region": "wide (Somali box excluded)",
        "train_years": train_years,
        "test_region": "somali",
        "test_years": test_years,
        "pooled": pooled,
        "within_month": [{k: v for k, v in w.items() if k != "per_month"}
                         for w in within],
        "beats_climatology": bool(wide_score > clim_score),
        "beats_somali_control": bool(wide_score > ctrl_score),
    }, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")


def load(features: list[str], neg_per_pos: int | None, train_end: int,
         seed: int = 42) -> pd.DataFrame:
    """Load all processed years, keeping every positive.

    Negatives are subsampled during loading, not after: the full table is
    ~22 million rows and will not fit comfortably in memory. Positives are
    never dropped, and the test split is left unsampled so reported metrics
    reflect the true class balance.
    """
    needed = sorted(set(features + [LABEL, "cell_id", "date", "month", "lat", "lon",
                                    "lat_idx", "lon_idx", "fishing_hours"])
                    - {"dist_coast_km"})   # derived after load, not stored yet
    frames = []
    rng = np.random.default_rng(seed)

    for year in available_years():
        path = config.PROCESSED_DIR / f"features_historical_{year}.parquet"
        try:
            df = pd.read_parquet(path, columns=needed)
        except Exception:
            # Older files may predate some columns; fall back to what exists.
            df = pd.read_parquet(path)
            df = df[[c for c in needed if c in df.columns]]
        df["date"] = pd.to_datetime(df["date"])
        df["year"] = df["date"].dt.year.astype("int16")

        # Test years keep EVERY row. Subsampling the test set would change the
        # base rate and inflate PR-AUC into meaninglessness. Note this keys off
        # the requested split, not config.TRAIN_END_YEAR.
        if neg_per_pos and year <= train_end:
            pos = df[df[LABEL] == 1]
            neg = df[df[LABEL] == 0]
            take = min(len(neg), len(pos) * neg_per_pos)
            if take < len(neg):
                keep = rng.choice(len(neg), size=take, replace=False)
                neg = neg.iloc[np.sort(keep)]
            df = pd.concat([pos, neg], ignore_index=True)
        frames.append(df)
        print(f"  {year}: {len(df):,} rows, {int(df[LABEL].sum()):,} positive")

    if not frames:
        sys.exit("No processed years. Run: python src/build_dataset.py")
    return pd.concat(frames, ignore_index=True)


# ------------------------------------------------------------------- train

def fit(train: pd.DataFrame, features: list[str], seed: int = 42):
    """Fit the classifier, holding out the last training year for calibration."""
    import xgboost as xgb

    # Calibration slice is the latest training year, kept temporally separate
    # so the calibrator is not fitted on data the model memorised.
    calib_year = train["year"].max()
    core = train[train["year"] < calib_year]
    calib = train[train["year"] == calib_year]
    if core.empty or calib.empty or core[LABEL].nunique() < 2:
        core, calib = train, None
        print("  [warn] not enough years to hold out a calibration slice")

    pos = int(core[LABEL].sum())
    neg = int((core[LABEL] == 0).sum())
    spw = max(1.0, neg / max(pos, 1))

    params = {
        "objective": "binary:logistic",
        "eval_metric": "aucpr",
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 20,
        "scale_pos_weight": spw,
        "nthread": 4,
        "seed": seed,
    }
    dtrain = xgb.DMatrix(core[features].astype("float32"), label=core[LABEL],
                         feature_names=features)
    watch = [(dtrain, "train")]
    dcalib = None
    if calib is not None:
        dcalib = xgb.DMatrix(calib[features].astype("float32"), label=calib[LABEL],
                             feature_names=features)
        watch.append((dcalib, "calib"))

    print(f"  training on {len(core):,} rows "
          f"({pos:,} positive, scale_pos_weight={spw:.1f})")
    booster = xgb.train(params, dtrain, num_boost_round=400, evals=watch,
                        early_stopping_rounds=30, verbose_eval=100)

    calibrator = None
    if calib is not None:
        from sklearn.isotonic import IsotonicRegression
        raw = booster.predict(dcalib)
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(raw, calib[LABEL].to_numpy())
        print(f"  calibrated on {len(calib):,} rows from {calib_year}")
    return booster, calibrator


def predict(booster, calibrator, df: pd.DataFrame,
            features: list[str]) -> np.ndarray:
    import xgboost as xgb
    d = xgb.DMatrix(df[features].astype("float32"), feature_names=features)
    raw = booster.predict(d)
    return calibrator.predict(raw) if calibrator is not None else raw


# ---------------------------------------------------------------- reporting

def feature_importance(booster, features: list[str]) -> list[dict]:
    gain = booster.get_score(importance_type="gain")
    total = sum(gain.values()) or 1.0
    rows = [{"feature": f, "gain_share": round(gain.get(f, 0.0) / total, 4)}
            for f in features]
    return sorted(rows, key=lambda r: -r["gain_share"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-end", type=int, default=config.TRAIN_END_YEAR)
    parser.add_argument("--neg-per-pos", type=int, default=20,
                        help="negatives kept per positive in TRAINING years "
                             "(test years are never subsampled)")
    parser.add_argument("--with-weather", action="store_true",
                        help="include wave/wind (see docstring - not advised)")
    parser.add_argument("--no-latlon", action="store_true",
                        help="drop lat/lon so the model cannot memorise places")
    parser.add_argument("--all-months", action="store_true",
                        help="keep fleet-dormant months (not advised)")
    parser.add_argument("--spatial-holdout", action="store_true",
                        help="also train excluding the Kismayo lat band and "
                             "test on it, to measure spatial transfer")
    parser.add_argument("--dry-run", action="store_true",
                        help="report data readiness and exit")
    parser.add_argument("--max-train-rows", type=int, default=2_500_000,
                        help="cap on training rows (positives always kept). "
                             "XGBoost holds the whole matrix in memory.")
    parser.add_argument("--transfer", action="store_true",
                        help="train on the wider Indian Ocean (Somali box "
                             "removed) and test on Somali waters, against a "
                             "Somali-trained control on the same features")
    args = parser.parse_args()

    if args.transfer:
        run_transfer(args)
        return

    features = list(FEATURES)
    if args.with_weather:
        features += WEATHER_FEATURES
    if args.no_latlon:
        features = [f for f in features if f not in ("lat", "lon")]

    years = available_years()
    train_years = [y for y in years if y <= args.train_end]
    test_years = [y for y in years if y > args.train_end]
    print(f"Processed years: {years}")
    print(f"  train <= {args.train_end}: {train_years}")
    print(f"  test  >  {args.train_end}: {test_years}")

    if args.dry_run:
        print("\nDry run only.")
        if not test_years:
            print("NOT READY: no test years built yet.")
        return
    if not train_years or not test_years:
        sys.exit("\nNeed processed years on both sides of the split.\n"
                 "Build them with: python src/build_dataset.py")

    print("\nLoading...")
    df = load(features, args.neg_per_pos, args.train_end)

    # dist_coast_km is static per cell and derived from the land mask, so it
    # can be filled in here rather than forcing a rebuild of every year.
    if "dist_coast_km" in features and "dist_coast_km" not in df.columns:
        import build_dataset
        import zones as zonelib
        lats, lons = build_dataset.build_grid()
        cells = df[["lat_idx", "lon_idx"]].drop_duplicates()
        cells["dist_coast_km"] = zonelib.distance_to_coast(cells, lats, lons).values
        plane = np.full((len(lats), len(lons)), np.nan, dtype="float32")
        plane[cells["lat_idx"].to_numpy(), cells["lon_idx"].to_numpy()] = \
            cells["dist_coast_km"].to_numpy()
        df["dist_coast_km"] = plane[df["lat_idx"].to_numpy(),
                                    df["lon_idx"].to_numpy()]
        print(f"  derived dist_coast_km (median "
              f"{np.nanmedian(df['dist_coast_km']):.0f} km)")

    missing = [f for f in features if f not in df.columns]
    if missing:
        sys.exit(f"Missing feature columns: {missing}\n"
                 "Rebuild with GEBCO present: python src/build_dataset.py")

    if not args.all_months:
        keep, active = baselines.fleet_active_months(df)
        print(f"fleet-active months {active}: keeping {keep.mean()*100:.1f}% of rows")
        df = df[keep]

    df["dist_coast_km"] = df.get("dist_coast_km", pd.Series(np.nan, index=df.index))

    train = df[df["year"] <= args.train_end]
    test = df[df["year"] > args.train_end]
    print(f"\ntrain {len(train):,} rows ({train[LABEL].mean()*100:.2f}% positive)")
    print(f"test  {len(test):,} rows ({test[LABEL].mean()*100:.2f}% positive)")

    print("\nFitting hotspot model...")
    booster, calibrator = fit(train, features)
    pred = predict(booster, calibrator, test, features)

    print("\nScoring baselines on the same test set...")
    base_depth = baselines.depth_only(train, test)
    base_clim = baselines.climatology(train, test)
    # The physics index is what the map currently ships, so it is measured
    # here too - shipping something untested would be its own dishonesty.
    base_phys = baselines.physics_index_baseline(train, test)

    scored = [("hotspot", pred), ("depth_only", base_depth),
              ("climatology", base_clim), ("physics_index", base_phys)]
    pooled = [baselines.evaluate(test[LABEL], p, n) for n, p in scored]
    within = [baselines.evaluate_within_month(test, p, name=n) for n, p in scored]

    print("\nPooled across months:")
    print(pd.DataFrame(pooled).to_string(index=False))
    print("\nWithin-month (the honest test):")
    print(pd.DataFrame([{k: v for k, v in w.items() if k != "per_month"}
                        for w in within]).to_string(index=False))

    hot = within[0]["pr_auc_within_month"]
    # The gate is the two reference baselines. The physics index is reported
    # alongside for comparison, not used as a pass condition.
    beat = {w["model"]: hot > w["pr_auc_within_month"]
            for w in within[1:] if w["model"] != "physics_index"}
    beats_all = all(beat.values())
    print(f"\nBeats depth_only: {beat['depth_only']}   "
          f"Beats climatology: {beat['climatology']}")

    phys = next((w for w in within if w["model"] == "physics_index"), None)
    if phys:
        print(f"\nPhysics index (what the map ships): within-month PR-AUC "
              f"{phys['pr_auc_within_month']} (lift {phys['lift_within_month']})")
        better = [w["model"] for w in within
                  if w["model"] not in ("physics_index",)
                  and w["pr_auc_within_month"] > phys["pr_auc_within_month"]]
        if better:
            print(f"  Beaten by: {better}. The shipped index is NOT validated "
                  f"as the best available ranking - say so in the pitch.")
        else:
            print("  The shipped index beats every alternative tested here.")

    print("\nFeature importance (gain share):")
    importance = feature_importance(booster, features)
    for row in importance[:10]:
        print(f"  {row['feature']:18s} {row['gain_share']*100:5.1f}%")

    # ---- spatial transfer
    spatial = None
    if args.spatial_holdout:
        lo, hi = SPATIAL_HOLDOUT_LAT
        print(f"\nSpatial holdout: training without lat {lo}..{hi} (Kismayo band)")
        in_band = train["lat"].between(lo, hi)
        test_band = test["lat"].between(lo, hi)
        if in_band.sum() == 0 or test_band.sum() == 0:
            print("  [skip] band empty on one side")
        else:
            b_model, b_cal = fit(train[~in_band], features)
            b_pred = predict(b_model, b_cal, test[test_band], features)
            spatial = baselines.evaluate_within_month(
                test[test_band], b_pred, name="hotspot_spatial_holdout")
            print(f"  within-month PR-AUC on unseen region: "
                  f"{spatial['pr_auc_within_month']} "
                  f"(lift {spatial['lift_within_month']})")
            print(f"  same-region reference: {hot} (lift {within[0]['lift_within_month']})")

    # ---- save
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(MODEL_PATH))
    meta = {
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "features": features,
        "label": LABEL,
        "train_years": train_years,
        "test_years": test_years,
        "weather_excluded": not args.with_weather,
        "fleet_dormant_months_excluded": not args.all_months,
        "metrics_pooled": pooled,
        "metrics_within_month": [{k: v for k, v in w.items() if k != "per_month"}
                                 for w in within],
        "per_month": within[0].get("per_month"),
        "spatial_holdout": spatial,
        "beats_baselines": beats_all,
        "feature_importance": importance,
        "caveats": [
            "Labels are AIS fishing effort: industrial and foreign vessels, "
            "not Somali artisanal boats.",
            "Not valid in fleet-dormant months (SW monsoon); the map falls "
            "back to the physics index and safety rules there.",
            "Predicts conditions attractive for fishing activity, not catch "
            "weight and not a probability of catching fish.",
        ],
    }
    META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nSaved {MODEL_PATH.name} and {META_PATH.name}")

    if beats_all:
        print("\nModel beats both baselines. predict.py will now rank zones "
              "with it instead of the physics index.")
    else:
        losers = [k for k, v in beat.items() if not v]
        print(f"\nMODEL DOES NOT BEAT: {losers}.")
        print("It is saved for inspection, but do not present it as validated. "
              "Keep ranking zones by the physics index until it wins.")
        sys.exit(2)


if __name__ == "__main__":
    main()
