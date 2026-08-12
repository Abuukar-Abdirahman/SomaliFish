"""Download satellite ocean data for Somali waters from Copernicus Marine.

STEP 1 of the pipeline. Idempotent: skips files that already exist.

Two dataset families (see src/config.py for why):
    --mode historical   long reanalysis records, for TRAINING (default)
    --mode nrt          near-real-time products, for the DAILY prediction

Sub-daily datasets (waves 3-hourly, wind hourly) are reduced to daily
statistics immediately after each chunk downloads, so raw hourly data never
accumulates on disk.

Prerequisites:
    pip install copernicusmarine
    copernicusmarine login   # once; free account at data.marine.copernicus.eu

Usage:
    python src/download_data.py --check              # verify IDs, no account needed
    python src/download_data.py                      # historical, START_DATE -> HISTORICAL_END
    python src/download_data.py --dataset sst        # one dataset
    python src/download_data.py --year 2022          # one year only
    python src/download_data.py --mode nrt --days 10 # recent data for prediction
"""

from __future__ import annotations

import argparse
import calendar
import datetime as dt
import sys
from typing import Iterator

import config


# --------------------------------------------------------------- date helpers

def year_ranges(start: str, end: str) -> Iterator[tuple[str, str, str]]:
    """Yield (label, start_iso, end_iso) tuples, one per calendar year."""
    start_d = dt.date.fromisoformat(start)
    end_d = dt.date.fromisoformat(end)
    for year in range(start_d.year, end_d.year + 1):
        a = max(start_d, dt.date(year, 1, 1))
        b = min(end_d, dt.date(year, 12, 31))
        if a <= b:
            yield str(year), a.isoformat(), b.isoformat()


def month_ranges(start: str, end: str) -> Iterator[tuple[str, str, str]]:
    """Yield (label, start_iso, end_iso) tuples, one per calendar month.

    Used for sub-daily datasets, where a whole year in one request is too
    large to hold in memory.
    """
    start_d = dt.date.fromisoformat(start)
    end_d = dt.date.fromisoformat(end)
    year, month = start_d.year, start_d.month
    while (year, month) <= (end_d.year, end_d.month):
        a = max(start_d, dt.date(year, month, 1))
        b = min(end_d, dt.date(year, month, calendar.monthrange(year, month)[1]))
        if a <= b:
            yield f"{year}-{month:02d}", a.isoformat(), b.isoformat()
        month += 1
        if month == 13:
            year, month = year + 1, 1


# ------------------------------------------------------------ daily reduction

def reduce_to_daily(path, name: str) -> None:
    """Collapse a sub-daily NetCDF to daily statistics, in place.

    Waves and wind arrive 3-hourly/hourly. For a daily fishing advisory we need
    the daily mean (typical conditions) and the daily max (the number that
    decides whether it is safe to go out). Keeping both, at daily resolution,
    is ~1/8 to ~1/24 the size of the raw file.
    """
    import xarray as xr

    with xr.open_dataset(path) as ds:
        daily = ds.resample(time="1D")
        out = daily.mean().rename({v: f"{v}_mean" for v in ds.data_vars})
        for v in ds.data_vars:
            out[f"{v}_max"] = daily.max()[v]
        out = out.load()

    tmp = path.with_suffix(".daily.nc")
    out.to_netcdf(tmp)
    out.close()
    path.unlink()
    tmp.rename(path)
    print(f"       reduced {name} to daily mean/max")


# ------------------------------------------------------------------- download

def download_dataset(name: str, spec: dict, start: str, end: str,
                     mode: str, overwrite: bool = False) -> tuple[int, int]:
    """Download one Copernicus dataset in chunks. Returns (ok, failed)."""
    try:
        import copernicusmarine
    except ImportError:
        sys.exit("copernicusmarine not installed. Run: pip install copernicusmarine")

    dataset_id = spec["dataset_id"]
    variables = spec["variables"]
    subdaily = name in config.SUBDAILY

    out_dir = config.region_dir(config.RAW_DIR) / mode / name
    out_dir.mkdir(parents=True, exist_ok=True)

    if mode == "nrt":
        # Clamp to what this product has actually published. Otherwise a
        # laggard (chlorophyll runs ~7-10 days behind) fails the whole request
        # instead of returning its most recent real data.
        bounds = dataset_time_bounds(dataset_id)
        if bounds:
            span = (dt.date.fromisoformat(end) - dt.date.fromisoformat(start)).days
            # Chlorophyll needs a long tail: it lags real time, and the 7- and
            # 14-day trailing means need history before the first target day.
            if name == "chl":
                span = max(span, config.NRT_CHL_LOOKBACK_DAYS)
                start = (dt.date.fromisoformat(end)
                         - dt.timedelta(days=span)).isoformat()
            if dt.date.fromisoformat(end) > bounds[1]:
                new_end = bounds[1]
                new_start = max(bounds[0], new_end - dt.timedelta(days=span))
                lag = (dt.date.today() - new_end).days
                print(f"       {name}: latest available is {new_end} "
                      f"({lag}d behind today) - using {new_start} -> {new_end}")
                start, end = new_start.isoformat(), new_end.isoformat()
        # NRT is a short rolling window refreshed daily; label by the window
        # itself so a later run does not collide with (and skip) an older file.
        chunks = [(f"{start}_{end}", start, end)]
    elif subdaily:
        chunks = month_ranges(start, end)
    else:
        chunks = year_ranges(start, end)
    ok = failed = 0

    for label, a, b in chunks:
        out_file = out_dir / f"{name}_{label}.nc"
        if out_file.exists() and not overwrite:
            print(f"[skip] {out_file.name} already exists")
            ok += 1
            continue
        print(f"[get ] {name} {label} ({a} -> {b})")
        # Model products carry 50 depth levels. Fishing happens at the surface,
        # and pulling every level would multiply the download by ~50x.
        depth_args = ({"minimum_depth": 0.0, "maximum_depth": 1.0}
                      if spec.get("surface_only") else {})
        try:
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=variables,
                **depth_args,
                minimum_latitude=config.LAT_MIN,
                maximum_latitude=config.LAT_MAX,
                minimum_longitude=config.LON_MIN,
                maximum_longitude=config.LON_MAX,
                start_datetime=f"{a}T00:00:00",
                end_datetime=f"{b}T23:59:59",
                output_directory=str(out_dir),
                output_filename=out_file.name,
                overwrite=True,
                disable_progress_bar=False,
            )
            if subdaily:
                reduce_to_daily(out_file, name)
            print(f"[ ok ] {out_file.name}")
            ok += 1
        except MemoryError as exc:
            # A whole year of a model product can exceed available memory.
            # Retry the same span month by month rather than leaving a hole in
            # the archive - a silently missing year produces a feature table
            # with whole columns absent.
            print(f"[mem ] {name} {label}: {exc}\n"
                  f"       retrying month by month")
            sub_ok = sub_fail = 0
            for sub_label, sa, sb in month_ranges(a, b):
                part = out_dir / f"{name}_{sub_label}.nc"
                if part.exists():
                    sub_ok += 1
                    continue
                try:
                    copernicusmarine.subset(
                        dataset_id=dataset_id, variables=variables, **depth_args,
                        minimum_latitude=config.LAT_MIN,
                        maximum_latitude=config.LAT_MAX,
                        minimum_longitude=config.LON_MIN,
                        maximum_longitude=config.LON_MAX,
                        start_datetime=f"{sa}T00:00:00",
                        end_datetime=f"{sb}T23:59:59",
                        output_directory=str(out_dir),
                        output_filename=part.name,
                        overwrite=True, disable_progress_bar=True)
                    sub_ok += 1
                except Exception as sub_exc:  # noqa: BLE001
                    sub_fail += 1
                    print(f"[FAIL] {name} {sub_label}: {sub_exc}")
            print(f"[ ok ] {name} {label}: {sub_ok} month(s) recovered"
                  + (f", {sub_fail} failed" if sub_fail else ""))
            ok += 1 if sub_ok and not sub_fail else 0
            failed += 1 if sub_fail else 0
        except Exception as exc:  # noqa: BLE001 - report and continue
            failed += 1
            print(f"[FAIL] {name} {label}: {type(exc).__name__}: {exc}")
            if "not found" in str(exc).lower() or "DatasetNotFound" in type(exc).__name__:
                print(f"       Dataset ID '{dataset_id}' did not resolve. The "
                      f"catalogue may have a newer version.\n"
                      f"       Run: python src/download_data.py --check")
            elif "credential" in str(exc).lower() or "401" in str(exc):
                print("       Not logged in. Run: copernicusmarine login")
    return ok, failed


# ---------------------------------------------------------------- check mode

def check_datasets(mode: str) -> int:
    """Verify every dataset ID resolves and report coverage. No account needed."""
    import copernicusmarine

    datasets = config.DATASETS_HISTORICAL if mode == "historical" else config.DATASETS_NRT
    print(f"Checking {len(datasets)} '{mode}' dataset IDs against the live catalogue...\n")
    bad = 0
    for name, spec in datasets.items():
        did = spec["dataset_id"]
        try:
            cat = copernicusmarine.describe(dataset_id=did, disable_progress_bar=True)
            if not cat.products:
                print(f"  MISSING  {name:7s} {did}")
                bad += 1
                continue
            avail = _coverage(cat, did)
            want = set(spec["variables"])
            missing = want - avail["vars"]
            status = "OK" if not missing else "VARS?"
            print(f"  {status:8s} {name:7s} {did}")
            print(f"           time {avail['time']}   (config says {spec['coverage']})")
            if missing:
                print(f"           MISSING VARIABLES: {sorted(missing)}")
                print(f"           available: {sorted(avail['vars'])[:12]}")
                bad += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR    {name:7s} {did} -> {type(exc).__name__}: {str(exc)[:90]}")
            bad += 1
    print()
    print("All dataset IDs valid." if not bad else f"{bad} dataset(s) need attention.")
    return bad


def dataset_time_bounds(dataset_id: str) -> tuple[dt.date, dt.date] | None:
    """Return the (first, last) date actually available for a dataset.

    Used to clamp near-real-time requests. Products publish on different
    delays -- gap-free chlorophyll typically runs ~7-10 days behind today --
    so asking for "the last 3 days" of everything fails on the laggards.
    """
    import copernicusmarine

    try:
        cat = copernicusmarine.describe(dataset_id=dataset_id, disable_progress_bar=True)
    except Exception:  # noqa: BLE001 - treat as unknown, caller falls back
        return None
    for product in cat.products:
        for ds in product.datasets:
            if ds.dataset_id != dataset_id:
                continue
            part = ds.versions[-1].parts[0]
            for svc in part.services:
                if str(svc.service_name) != "arco-geo-series":
                    continue
                for var in svc.variables:
                    for coord in var.coordinates:
                        if coord.coordinate_id == "time" and coord.minimum_value is not None:
                            to_date = (lambda ms: dt.datetime.fromtimestamp(
                                ms / 1000, dt.timezone.utc).date())
                            return to_date(coord.minimum_value), to_date(coord.maximum_value)
    return None


def _coverage(catalogue, dataset_id: str) -> dict:
    """Pull variable names and time range out of a describe() result."""
    def iso(ms):
        if ms is None:
            return "?"
        try:
            return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).date().isoformat()
        except (OSError, ValueError, OverflowError):
            return str(ms)

    for product in catalogue.products:
        for ds in product.datasets:
            if ds.dataset_id != dataset_id:
                continue
            part = ds.versions[-1].parts[0]
            services = [s for s in part.services
                        if str(s.service_name) == "arco-geo-series"]
            if not services:
                continue
            svc = services[0]
            tmin = tmax = None
            for var in svc.variables:
                for coord in var.coordinates:
                    if coord.coordinate_id == "time" and coord.minimum_value is not None:
                        tmin, tmax = coord.minimum_value, coord.maximum_value
            return {"vars": {v.short_name for v in svc.variables},
                    "time": f"{iso(tmin)} -> {iso(tmax)}"}
    return {"vars": set(), "time": "?"}


# ------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Copernicus Marine ocean data for the Somali EEZ.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[-1])
    parser.add_argument("--mode", choices=["historical", "nrt"], default="historical",
                        help="historical = training data (default); nrt = today's data")
    parser.add_argument("--dataset", help="download only this dataset key")
    parser.add_argument("--year", type=int, help="download only this year")
    parser.add_argument("--days", type=int,
                        help="nrt mode: download the last N days up to today")
    parser.add_argument("--start", default=config.START_DATE)
    parser.add_argument("--end", default=None,
                        help="default: HISTORICAL_END (historical) or today (nrt)")
    parser.add_argument("--check", action="store_true",
                        help="verify dataset IDs against the catalogue and exit "
                             "(no Copernicus account needed)")
    parser.add_argument("--overwrite", action="store_true",
                        help="re-download files that already exist")
    parser.add_argument("--region", choices=list(config.REGIONS), default="somali",
                        help="'wide' downloads the western Indian Ocean "
                             "training region into its own directory")
    parser.add_argument("--training-only", action="store_true",
                        help="only the datasets the hotspot model uses "
                             f"({', '.join(config.TRAINING_DATASETS)}) - skips "
                             "waves and wind, which are excluded from the model")
    args = parser.parse_args()

    config.set_region(args.region)

    if args.check:
        sys.exit(1 if check_datasets(args.mode) else 0)

    datasets = (config.DATASETS_HISTORICAL if args.mode == "historical"
                else config.DATASETS_NRT)

    # ---- resolve the date window
    if args.mode == "historical":
        start = args.start
        end = args.end or config.HISTORICAL_END
    else:
        today = dt.date.today()
        end = args.end or today.isoformat()
        start = args.start if args.end or args.start != config.START_DATE else None
        if args.days:
            start = (today - dt.timedelta(days=args.days)).isoformat()
        elif start is None:
            start = (today - dt.timedelta(days=14)).isoformat()
    if args.year:
        start, end = f"{args.year}-01-01", f"{args.year}-12-31"

    if args.dataset:
        if args.dataset not in datasets:
            sys.exit(f"Unknown dataset '{args.dataset}' for mode '{args.mode}'. "
                     f"Choose from: {', '.join(datasets)}")
        items = [(args.dataset, datasets[args.dataset])]
    else:
        items = list(datasets.items())
    if args.training_only:
        items = [(n, s) for n, s in items if n in config.TRAINING_DATASETS]

    print(f"Region: {args.region} "
          f"({config.REGIONS[args.region]['note']})")
    print(f"Mode: {args.mode}   Window: {start} -> {end}")
    print(f"Area: lat {config.LAT_MIN}..{config.LAT_MAX}, "
          f"lon {config.LON_MIN}..{config.LON_MAX}")
    print(f"Output: {config.region_dir(config.RAW_DIR) / args.mode}\n")

    total_ok = total_failed = 0
    for name, spec in items:
        ok, failed = download_dataset(name, spec, start, end, args.mode, args.overwrite)
        total_ok += ok
        total_failed += failed

    print(f"\n{total_ok} chunk(s) ready, {total_failed} failed.")
    if total_failed:
        print("Some downloads failed - see messages above. Nothing was faked; "
              "re-run to retry (existing files are skipped).")
        sys.exit(1)
    print("Next: GEBCO depth (docs/DATA_SOURCES.md §2), then src/build_dataset.py")


if __name__ == "__main__":
    main()
