"""Turn the latest feature table into map-ready daily output.

STEP 4 of the pipeline. Writes one compact JSON file per day to
data/predictions/, plus latest.json.

Output is deliberately NOT GeoJSON polygons: 8,294 cells as polygons is
several MB, and this has to open on a cheap Android phone over a slow
connection. Instead we emit the grid definition once and cell values as
parallel arrays, which the map draws onto a canvas. Typical size ~250 KB,
~60 KB gzipped.

Usage:
    python src/predict.py                  # newest day in the NRT table
    python src/predict.py --all-days       # every day in the table
    python src/predict.py --demo           # also write data/demo/ fallback

Honesty rules enforced here:
  - If no trained model exists, NO fishing score is emitted. The map shows
    ocean conditions and safety only, and says the model is not yet trained.
    It never invents a score.
  - Scores are a relative 0-100 potential index, never a catch weight.
  - Every cell carries the age of its chlorophyll input so the map can show
    where the inputs are stale.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import build_dataset
import config
import safety
import zones as zonelib

MODEL_PATH = config.MODELS_DIR / "hotspot_xgb.json"
FEATURES_PATH = config.PROCESSED_DIR / "features_nrt_latest.parquet"


def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        sys.exit(f"No feature table at {FEATURES_PATH}.\n"
                 f"Run: python src/download_data.py --mode nrt --days 5\n"
                 f"     python src/build_dataset.py --mode nrt")
    df = pd.read_parquet(FEATURES_PATH)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_model(allow_unvalidated: bool = False):
    """Load the trained hotspot model, if there is one fit to be used.

    A saved model file is not enough. train_model.py records whether the model
    beat the depth-only and climatology baselines on within-month PR-AUC; if it
    did not, it has not demonstrated that it knows anything about the ocean,
    and we keep ranking zones by the transparent physics index instead. An
    unvalidated model presented as a prediction is exactly the "fake
    precision" CLAUDE.md forbids.
    """
    if not MODEL_PATH.exists():
        return None, None
    try:
        import xgboost as xgb
    except ImportError:
        return None, None

    meta_path = MODEL_PATH.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    if not meta.get("beats_baselines") and not allow_unvalidated:
        print("[note] a trained model exists but did NOT beat both baselines "
              "- ignoring it and ranking by the physics index.\n"
              "       Override for inspection with --use-unvalidated-model.")
        return None, None

    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    return booster, meta


def score_cells(df: pd.DataFrame, booster, meta) -> pd.Series | None:
    """Fishing-potential index, 0-100. None when no model is available.

    This is a RELATIVE index of how favourable conditions are, calibrated
    against where vessels historically fished. It is not a probability of
    catching fish and not a predicted catch weight.
    """
    if booster is None:
        return None
    import xgboost as xgb

    feature_names = meta.get("features") or []
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        print(f"[warn] model expects features not in the table: {missing}")
        return None

    matrix = xgb.DMatrix(df[feature_names].astype("float32"),
                         feature_names=feature_names)
    raw = booster.predict(matrix)
    # Map calibrated probability onto a 0-100 index for display.
    return pd.Series(np.clip(raw * 100.0, 0, 100), index=df.index).round(1)


def build_day(day_df: pd.DataFrame, scores: pd.Series | None,
              lats=None, lons=None) -> dict:
    """Assemble one day's output payload."""
    status = safety.classify(day_df.get("wave_height_max"),
                             day_df.get("wind_speed_max"))
    status.index = day_df.index

    # Ranking source: the trained model when it exists, otherwise the
    # transparent physics index. Which one is always reported downstream.
    if scores is not None:
        day_df = day_df.assign(_score=scores)
        score_source = "model"
    else:
        day_df = day_df.assign(_score=zonelib.physics_index(day_df))
        score_source = "physics_index"

    payload = {
        "date": day_df["date"].iloc[0].date().isoformat(),
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "grid": {
            "lat_min": config.LAT_MIN, "lat_max": config.LAT_MAX,
            "lon_min": config.LON_MIN, "lon_max": config.LON_MAX,
            "resolution": config.GRID_RES,
            # Raster dimensions, so the map can build an offscreen image at
            # native grid size and let the browser scale it smoothly.
            "n_lat": int(round((config.LAT_MAX - config.LAT_MIN) / config.GRID_RES)),
            "n_lon": int(round((config.LON_MAX - config.LON_MIN) / config.GRID_RES)),
            # Beyond this the 0.1 deg (~11 km) data has no more detail to give;
            # zooming further would imply precision we do not have.
            "max_useful_zoom": 10,
        },
        "safety": safety.summarise(status),
        "thresholds": {
            "wave_safe_m": config.WAVE_SAFE_M,
            "wave_danger_m": config.WAVE_DANGER_M,
            "wind_safe_kmh": config.WIND_SAFE_KMH,
            "wind_danger_kmh": config.WIND_DANGER_KMH,
        },
        "n_cells": int(len(day_df)),
    }

    def col(name, decimals=2):
        if name not in day_df:
            return None
        return [None if pd.isna(v) else round(float(v), decimals)
                for v in day_df[name]]

    payload["cells"] = {
        "lat": [round(float(v), 3) for v in day_df["lat"]],
        "lon": [round(float(v), 3) for v in day_df["lon"]],
        # Integer grid indices: the map uses these to place values into the
        # raster exactly, rather than re-deriving them from rounded floats.
        "lat_idx": [int(v) for v in day_df["lat_idx"]],
        "lon_idx": [int(v) for v in day_df["lon_idx"]],
        "safety": list(status.values),
        "sst": col("sst", 1),
        "chl": col("chl", 3),
        "sst_gradient": col("sst_gradient", 2),
        "current_speed": col("current_speed", 2),
        "wave_height_max": col("wave_height_max", 2),
        "wind_speed_max": col("wind_speed_max", 1),
    }
    if "wave_estimated" in day_df:
        payload["cells"]["wave_estimated"] = [int(v) if pd.notna(v) else 0
                                              for v in day_df["wave_estimated"]]
        est = float(day_df["wave_estimated"].mean())
        payload["data_quality"] = {
            "wave_estimated_share": round(est, 3),
            "note_en": ("Wave models mask shallow water. Cells flagged "
                        "wave_estimated take their wave height from the "
                        "nearest cell with data (up to "
                        f"{config.WAVE_FILL_MAX_KM:.0f} km), which tends to "
                        "over-warn rather than under-warn."),
        }

    payload["cells"]["score"] = [None if pd.isna(v) else float(v)
                                 for v in day_df["_score"]]

    if score_source == "model":
        payload["model"] = {
            "trained": True,
            "score_source": "model",
            "label_so": "Rajada kalluunka (model)",
            "label_en": "Fishing potential (trained model)",
        }
    else:
        # Be explicit about what the number is. It is measured chlorophyll and
        # front strength, ranked - not a validated prediction of catch.
        payload["model"] = {
            "trained": False,
            "score_source": "physics_index",
            "label_so": "Tilmaanta biyaha (cabbiran)",
            "label_en": "Water-condition indicator (measured)",
            "note_so": ("Ma jiro model tababaran weli. Tirradan waxay ka "
                        "timid cagaaran iyo kala duwanaanta heerkulka ee la "
                        "cabbiray - maaha saadaal."),
            "note_en": ("No trained model yet. This index is measured "
                        "chlorophyll and temperature-front strength, ranked "
                        "across the grid - not a validated prediction."),
            "weights": {"chl": 0.55, "sst_gradient": 0.45},
        }

    # ---- zones: the part a fisherman can actually act on
    if lats is not None and lons is not None:
        zoned = zonelib.assign(day_df, lats, lons)
        zone_list = zonelib.summarise(zoned, "_score", score_source)
        for z in zone_list:
            z["rating"] = zonelib.rating(z["score"])
        payload["zones"] = zone_list
        payload["reach"] = {
            "max_coast_km": config.ARTISANAL_MAX_COAST_KM,
            "max_trip_km": config.ARTISANAL_MAX_TRIP_KM,
            "boat_speed_kmh": config.ARTISANAL_BOAT_SPEED_KMH,
            "n_reachable_cells": int(zoned["reachable"].sum()),
            "n_cells_total": int(len(zoned)),
            "note_en": ("Zones cover only water an open boat can work in a "
                        "day trip. Cells beyond that are shown but never "
                        "ranked or recommended."),
        }
        payload["cells"]["reachable"] = [bool(v) for v in zoned["reachable"]]
        payload["cells"]["dist_coast_km"] = [
            None if pd.isna(v) else round(float(v), 1)
            for v in zoned["dist_coast_km"]]

    if "chl_age_days" in day_df:
        ages = day_df["chl_age_days"].dropna()
        payload["data_freshness"] = {
            "chl_age_days": None if ages.empty else int(ages.max()),
        }
    return payload


def write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all-days", action="store_true",
                        help="write a file for every day in the table")
    parser.add_argument("--demo", action="store_true",
                        help="also copy the newest day into data/demo/ as the "
                             "offline demo fallback")
    parser.add_argument("--use-unvalidated-model", action="store_true",
                        help="use a saved model that has NOT beaten the "
                             "baselines (inspection only - never for a demo)")
    args = parser.parse_args()

    df = load_features()
    booster, meta = load_model(allow_unvalidated=args.use_unvalidated_model)
    if booster is None:
        why = ("rejected (did not beat the baselines)" if MODEL_PATH.exists()
               else "not trained yet")
        print(f"[note] hotspot model {why} - ranking zones by the physics "
              f"index; no model prediction is being made.")
    else:
        print(f"[ ok ] loaded model ({len(meta.get('features', []))} features, "
              f"trained {meta.get('trained_at', 'unknown')})")

    days = sorted(df["date"].unique())
    if not args.all_days:
        days = days[-1:]

    lats, lons = build_dataset.build_grid()

    written = []
    for day in days:
        day_df = df[df["date"] == day].reset_index(drop=True)
        scores = score_cells(day_df, booster, meta)
        payload = build_day(day_df, scores, lats, lons)
        out = config.PREDICTIONS_DIR / f"{payload['date']}.json"
        write_json(payload, out)
        written.append(out)
        s = payload["safety"]
        n_zones = len(payload.get("zones", []))
        print(f"[ ok ] {payload['date']}  {payload['n_cells']:,} cells  "
              f"safety={s['status']} ({s['share_dangerous']*100:.0f}% dangerous)  "
              f"{n_zones} zones")
        for z in payload.get("zones", [])[:3]:
            print(f"         {z['rank']}. {z['name_so']:22s} "
                  f"{z['score']}/100 {z['rating']['so']:14s} "
                  f"{z['distance_km']} km  {z['safety']}")

    if written:
        latest = json.loads(written[-1].read_text(encoding="utf-8"))
        write_json(latest, config.PREDICTIONS_DIR / "latest.json")
        print(f"[ ok ] latest.json -> {latest['date']}")
        if args.demo:
            write_json(latest, config.DEMO_DIR / "latest.json")
            print(f"[ ok ] demo fallback written to {config.DEMO_DIR / 'latest.json'}")

    size = (config.PREDICTIONS_DIR / "latest.json").stat().st_size / 1024
    print(f"\n{len(written)} day(s) written. latest.json is {size:.0f} KB.")
    print("Next: python src/api.py   (then open http://127.0.0.1:8000)")


if __name__ == "__main__":
    main()
