"""Download Global Fishing Watch apparent fishing effort for Somali waters.

STEP 1b of the pipeline. These are the v1 training LABELS: where AIS-carrying
vessels actually chose to fish, day by day, on the same 0.1 degree grid as the
ocean features.

Idempotent: one Parquet file per month, existing months are skipped.

Prerequisites:
    A free API token from https://globalfishingwatch.org/our-apis/
    stored in a .env file at the project root as:
        GFW_API_TOKEN=<token>

Usage:
    python src/download_gfw.py                     # START_DATE -> HISTORICAL_END
    python src/download_gfw.py --year 2023
    python src/download_gfw.py --check             # verify the token works
    python src/download_gfw.py --summary           # who is fishing out there

KNOWN LIMITATION, restated because it matters: GFW effort is derived from AIS
transponders, which are carried by industrial and foreign vessels. Somali
artisanal boats almost never carry AIS and are therefore close to invisible
here. A model trained on these labels learns "where industrial vessels fish",
which we treat as a proxy for productive water -- not as Somali catch. Say so
in the outputs and the pitch. See docs/DATA_SOURCES.md section 3.
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

import config

API_URL = "https://gateway.api.globalfishingwatch.org/v3/4wings/report"
EFFORT_DATASET = "public-global-fishing-effort:latest"

# Columns we keep. The API also returns vessel names, IMO and MMSI; we drop
# those - we need fishing locations, not a surveillance record of named boats.
KEEP = ["date", "lat", "lon", "hours", "flag", "geartype", "vesselType"]


def load_token() -> str:
    """Read GFW_API_TOKEN from the environment or the .env file."""
    import os

    token = os.environ.get("GFW_API_TOKEN")
    if token:
        return token.strip()

    env_file = config.ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("GFW_API_TOKEN="):
                return line.split("=", 1)[1].strip()

    sys.exit(
        "No GFW API token found.\n"
        "  1. Get a free token: https://globalfishingwatch.org/our-apis/\n"
        f"  2. Save it to {env_file} as:  GFW_API_TOKEN=<token>\n"
        "  (.env is already in .gitignore)")


def somali_eez_polygon() -> dict:
    """Our bounding box as a GeoJSON polygon, for the API region filter."""
    return {
        "type": "Polygon",
        "coordinates": [[
            [config.LON_MIN, config.LAT_MIN],
            [config.LON_MAX, config.LAT_MIN],
            [config.LON_MAX, config.LAT_MAX],
            [config.LON_MIN, config.LAT_MAX],
            [config.LON_MIN, config.LAT_MIN],
        ]],
    }


def fetch_month(token: str, year: int, month: int,
                retries: int = 3) -> pd.DataFrame:
    """Fetch one month of daily, 0.1 degree fishing effort."""
    start = dt.date(year, month, 1)
    end = start + dt.timedelta(days=calendar.monthrange(year, month)[1])

    params = {
        "spatial-resolution": "LOW",        # LOW = 0.1 deg, matching our grid
        "temporal-resolution": "DAILY",
        "datasets[0]": EFFORT_DATASET,
        "date-range": f"{start.isoformat()},{end.isoformat()}",
        "format": "JSON",
    }
    headers = {"Authorization": f"Bearer {token}",
               "Content-Type": "application/json"}
    body = {"geojson": somali_eez_polygon()}

    last_error = None
    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, params=params, headers=headers,
                                 json=body, timeout=300)
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(3 * (attempt + 1))
            continue

        if resp.status_code == 429:            # rate limited
            wait = int(resp.headers.get("Retry-After", 20))
            print(f"       rate limited, waiting {wait}s")
            time.sleep(wait)
            continue
        if resp.status_code == 401:
            sys.exit("GFW rejected the token (401). It may be expired or "
                     "revoked - generate a new one and update .env")
        if not resp.ok:
            last_error = f"HTTP {resp.status_code}: {resp.text[:200]}"
            time.sleep(3 * (attempt + 1))
            continue

        payload = resp.json()
        entries = payload.get("entries") or []
        rows: list[dict] = []
        for entry in entries:
            for value in entry.values():
                if isinstance(value, list):
                    rows.extend(value)
        if not rows:
            return pd.DataFrame(columns=KEEP)

        df = pd.DataFrame(rows)
        for col in KEEP:
            if col not in df:
                df[col] = pd.NA
        return df[KEEP]

    raise RuntimeError(f"failed after {retries} attempts: {last_error}")


def month_list(start: str, end: str) -> list[tuple[int, int]]:
    a, b = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    out, year, month = [], a.year, a.month
    while (year, month) <= (b.year, b.month):
        out.append((year, month))
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


def to_grid_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-vessel rows to fishing hours per grid cell per day."""
    if df.empty:
        return pd.DataFrame(columns=["date", "lat", "lon", "fishing_hours",
                                     "n_vessels"])
    df = df.copy()
    # Snap to our 0.1 deg cell centres so the join with ocean features is exact.
    df["lat"] = ((df["lat"] / config.GRID_RES).round() * config.GRID_RES
                 + config.GRID_RES / 2).round(4)
    df["lon"] = ((df["lon"] / config.GRID_RES).round() * config.GRID_RES
                 + config.GRID_RES / 2).round(4)
    grouped = (df.groupby(["date", "lat", "lon"], as_index=False)
                 .agg(fishing_hours=("hours", "sum"),
                      n_vessels=("hours", "size")))
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--start", default=config.START_DATE)
    parser.add_argument("--end", default=config.HISTORICAL_END)
    parser.add_argument("--year", type=int, help="download only this year")
    parser.add_argument("--check", action="store_true",
                        help="verify the token against the API and exit")
    parser.add_argument("--summary", action="store_true",
                        help="summarise what has been downloaded and exit")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--region", choices=list(config.REGIONS), default="somali",
                        help="'wide' pulls the western Indian Ocean training "
                             "region into its own directory")
    args = parser.parse_args()

    config.set_region(args.region)
    out_dir = config.region_dir(config.RAW_DIR) / "gfw"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.region != "somali":
        print(f"Region: {args.region} "
              f"(lat {config.LAT_MIN}..{config.LAT_MAX}, "
              f"lon {config.LON_MIN}..{config.LON_MAX})")

    if args.summary:
        summarise(out_dir)
        return

    token = load_token()

    if args.check:
        df = fetch_month(token, 2023, 1)
        print(f"Token works. January 2023 returned {len(df):,} vessel-day rows "
              f"in the Somali EEZ box.")
        if not df.empty:
            print("Top flags:",
                  ", ".join(f"{k}={v}" for k, v in
                            df["flag"].value_counts().head(5).items()))
        return

    start, end = args.start, args.end
    if args.year:
        start, end = f"{args.year}-01-01", f"{args.year}-12-31"

    months = month_list(start, end)
    print(f"GFW fishing effort: {len(months)} month(s), {start} -> {end}")
    print(f"Area: lat {config.LAT_MIN}..{config.LAT_MAX}, "
          f"lon {config.LON_MIN}..{config.LON_MAX}\n")

    ok = failed = skipped = 0
    for year, month in months:
        out_file = out_dir / f"gfw_{year}-{month:02d}.parquet"
        if out_file.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            df = fetch_month(token, year, month)
            df.to_parquet(out_file, index=False)
            print(f"[ ok ] {year}-{month:02d}  {len(df):5,} rows")
            ok += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[FAIL] {year}-{month:02d}: {exc}")
            failed += 1
        time.sleep(0.5)     # be polite to a free API

    print(f"\n{ok} downloaded, {skipped} already present, {failed} failed.")
    if failed:
        print("Re-run to retry the failed months (existing files are skipped).")
        sys.exit(1)
    summarise(out_dir)


def summarise(out_dir: Path) -> None:
    """Report coverage and who is actually fishing in Somali waters."""
    files = sorted(out_dir.glob("gfw_*.parquet"))
    if not files:
        print("Nothing downloaded yet.")
        return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"\n{len(files)} month(s), {len(df):,} vessel-day rows, "
          f"{df['hours'].sum():,.0f} total apparent fishing hours")
    print(f"dates: {df['date'].min()} -> {df['date'].max()}")

    cells = to_grid_cells(df)
    print(f"{len(cells):,} cell-days with fishing, "
          f"{cells.groupby(['lat', 'lon']).ngroups:,} distinct cells")

    print("\nApparent fishing hours by flag (top 10):")
    by_flag = (df.groupby(df["flag"].fillna("(unknown)"))["hours"]
                 .sum().sort_values(ascending=False))
    total = by_flag.sum()
    for flag, hours in by_flag.head(10).items():
        print(f"  {flag:12s} {hours:10,.0f} h  ({hours / total * 100:4.1f}%)")
    if "SOM" in by_flag.index:
        som = by_flag["SOM"] / total * 100
        print(f"\n  Somali-flagged: {som:.2f}% of apparent fishing hours.")
    else:
        print("\n  Somali-flagged vessels: ABSENT from this data entirely.")
    print("  This is the AIS limitation, not an error. Somali artisanal boats\n"
          "  do not carry transponders. Report it honestly in the pitch.")


if __name__ == "__main__":
    main()
