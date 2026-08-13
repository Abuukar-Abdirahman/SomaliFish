"""Merge downloaded ocean data into the model training table.

STEP 2 of the pipeline. One row = one ocean grid cell x one day.

Reads the NetCDF files written by download_data.py, regrids every variable onto
the common 0.1 degree Somali EEZ grid, derives the physical features the model
needs, and writes one Parquet file per year to data/processed/.

Usage:
    python src/build_dataset.py                     # all years, historical
    python src/build_dataset.py --year 2023         # one year
    python src/build_dataset.py --mode nrt          # today's data, for predict.py
    python src/build_dataset.py --list              # what is downloaded so far

Features produced (see docs/ARCHITECTURE.md):
    sst, sst_gradient, chl, chl_lag7, chl_lag14, current_speed,
    wave_height, wave_height_max, wind_speed, wind_speed_max,
    month_sin, month_cos, lat, lon

Depth and distance-to-coast come from GEBCO and are added by a separate step
(they are static, not per-day) -- see add_bathymetry() below.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import config


# ------------------------------------------------------------------- the grid

def build_grid() -> tuple[np.ndarray, np.ndarray]:
    """Return the (lat, lon) axes of the common 0.1 degree analysis grid.

    Cell centres, not edges: a cell labelled 5.05 covers 5.00-5.10.
    """
    lats = np.arange(config.LAT_MIN, config.LAT_MAX, config.GRID_RES) + config.GRID_RES / 2
    lons = np.arange(config.LON_MIN, config.LON_MAX, config.GRID_RES) + config.GRID_RES / 2
    return np.round(lats, 4), np.round(lons, 4)


def cell_index(lat, lon):
    """Map latitude/longitude onto integer grid indices.

    Always join on these, never on floating-point coordinates: the feature
    table stores lat/lon as float32 to save memory, and float32(-1.85) is not
    equal to float64(-1.85), so a float merge silently drops most rows.
    """
    lat_idx = np.rint((np.asarray(lat, dtype="float64") - config.LAT_MIN
                       - config.GRID_RES / 2) / config.GRID_RES).astype("int32")
    lon_idx = np.rint((np.asarray(lon, dtype="float64") - config.LON_MIN
                       - config.GRID_RES / 2) / config.GRID_RES).astype("int32")
    return lat_idx, lon_idx


def format_cell_id(lat_idx, lon_idx) -> pd.Series:
    """Stable string ID from integer grid indices, e.g. 'c_0095_0104'."""
    return ("c_" + pd.Series(lat_idx).astype(str).str.zfill(4)
            + "_" + pd.Series(lon_idx).astype(str).str.zfill(4))


def cell_ids(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """Stable string IDs for grid cells, e.g. 'c_0512_0447'.

    Index-based rather than coordinate-based so the ID never changes if the
    bounding box is widened later.
    """
    ii, jj = np.meshgrid(np.arange(len(lats)), np.arange(len(lons)), indexing="ij")
    return np.char.add(np.char.add(np.char.add(
        "c_", np.char.zfill(ii.astype(str), 4)), "_"),
        np.char.zfill(jj.astype(str), 4))


# ------------------------------------------------------------------- loading

def _open_many(pattern: str) -> xr.Dataset | None:
    """Open every NetCDF matching a glob and concatenate along time."""
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    ds = xr.open_mfdataset(files, combine="by_coords", engine="netcdf4")
    # Model products keep a length-1 depth axis after the surface subset.
    if "depth" in ds.dims:
        ds = ds.isel(depth=0, drop=True)
    return ds


def _to_grid(ds: xr.Dataset, lats: np.ndarray, lons: np.ndarray,
             time_chunk: int = 40) -> xr.Dataset:
    """Interpolate a dataset onto the common grid.

    Linear interpolation. Every source is coarser than or close to 0.1 degrees,
    so this is downsampling or mild resampling, not invention of detail.

    Done in time slices: a full year of 4 km chlorophyll over the wide region
    is a single 250 x 160,600 float64 allocation, which does not fit. Chunking
    costs nothing and makes the wide grid possible.
    """
    if "time" not in ds.dims or ds.sizes["time"] <= time_chunk:
        return ds.interp(latitude=lats, longitude=lons, method="linear")

    pieces = []
    n = ds.sizes["time"]
    for start in range(0, n, time_chunk):
        piece = ds.isel(time=slice(start, start + time_chunk))
        pieces.append(piece.interp(latitude=lats, longitude=lons,
                                   method="linear").load())
    return xr.concat(pieces, dim="time")


def _daily(ds: xr.Dataset) -> xr.Dataset:
    """Snap the time axis to whole days so sources can be joined on date."""
    return ds.assign_coords(time=ds.time.dt.floor("D"))


# ------------------------------------------------------------------ features

def sst_gradient(sst: xr.DataArray) -> xr.DataArray:
    """Magnitude of the horizontal SST gradient, in degrees C per 100 km.

    Ocean fronts -- sharp temperature boundaries -- concentrate prey and are
    the classic Potential Fishing Zone signal used by India's INCOIS
    advisories. This is the single most important engineered feature.
    """
    # One degree of latitude is ~111.32 km everywhere; one degree of longitude
    # shrinks by cos(latitude), which matters across our 14.5 degree span.
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(sst.latitude))

    d_dlat = sst.differentiate("latitude") / km_per_deg_lat      # degC per km
    d_dlon = sst.differentiate("longitude") / km_per_deg_lon     # degC per km

    return np.sqrt(d_dlat ** 2 + d_dlon ** 2) * 100.0            # degC / 100 km


def fill_from_nearest(da: xr.DataArray, max_km: float) -> tuple[xr.DataArray, xr.DataArray]:
    """Fill coastal gaps from the nearest valid cell, within max_km.

    Wave models mask shallow water, so 77% of cells within 15 km of the Somali
    coast have no wave height -- exactly where artisanal boats work. Leaving
    them blank makes the safety layer useless to its actual users.

    Filling from the nearest offshore cell is deliberately conservative: waves
    offshore are generally larger than inshore, so the estimate errs toward
    over-warning. For a safety system that is the correct direction to be
    wrong. Cells filled this way are flagged so the map can mark them.

    Returns (filled, was_estimated).
    """
    from scipy.ndimage import distance_transform_edt

    dy = config.GRID_RES * 111.32
    dx = config.GRID_RES * 111.32 * float(np.cos(np.deg2rad(
        float(da.latitude.mean()))))

    values = da.values.copy()
    estimated = np.zeros_like(values, dtype=bool)
    has_time = "time" in da.dims
    slices = range(values.shape[0]) if has_time else [None]

    for t in slices:
        plane = values[t] if t is not None else values
        invalid = np.isnan(plane)
        if not invalid.any() or invalid.all():
            continue
        dist, (ii, jj) = distance_transform_edt(
            invalid, sampling=(dy, dx), return_indices=True)
        within = invalid & (dist <= max_km)
        plane[within] = plane[ii[within], jj[within]]
        if t is not None:
            values[t] = plane
            estimated[t] = within
        else:
            estimated = within

    filled = da.copy(data=values)
    flag = da.copy(data=estimated).astype("int8").rename(f"{da.name}_estimated")
    return filled, flag


def carry_chlorophyll_forward(ds: xr.Dataset) -> xr.Dataset:
    """Fill chlorophyll gaps from the most recent observation, and record age.

    Chlorophyll is published ~7-10 days behind real time and also has cloud
    gaps, so the most recent days never have their own value. Rather than drop
    those days (which would mean no prediction for today) or silently pretend
    the data is current, we carry the last real observation forward and expose
    `chl_age_days` as a feature. The model can learn to discount stale
    chlorophyll, and the map can show honest confidence.

    Beyond CHL_MAX_CARRY_FORWARD_DAYS the value is left NaN: an old plankton
    reading is not evidence about today.
    """
    chl_vars = [v for v in ds.data_vars if str(v).startswith("chl")]
    if not chl_vars or "time" not in ds.dims:
        return ds

    limit = config.CHL_MAX_CARRY_FORWARD_DAYS
    observed = np.asarray(ds["chl"].notnull().any(dim=("latitude", "longitude")))
    times = ds.time.values
    if not observed.any():
        print("  [warn] chlorophyll has no observed days in this window")
        return ds

    # For each day, the index of the most recent day that has real data.
    # (A plain index gather, rather than DataArray.ffill, which would pull in
    # the optional bottleneck dependency.)
    src_idx = np.empty(len(times), dtype=int)
    last = -1
    for i, has_data in enumerate(observed):
        if has_data:
            last = i
        src_idx[i] = last
    leading_gap = src_idx < 0          # days before the first observation
    src_idx[leading_gap] = 0

    age_days = np.where(
        leading_gap, np.nan,
        (times - times[src_idx]) / np.timedelta64(1, "D")).astype("float32")
    too_old = np.isnan(age_days) | (age_days > limit)

    for var in chl_vars:
        gathered = ds[var].isel(time=src_idx).assign_coords(time=ds.time)
        # An old plankton reading is not evidence about today.
        ds[var] = gathered.where(~xr.DataArray(too_old, coords={"time": ds.time},
                                               dims=("time",)))
    ds["chl_age_days"] = xr.DataArray(age_days, coords={"time": ds.time},
                                      dims=("time",))

    filled = int(((age_days > 0) & (age_days <= limit)).sum())
    if filled:
        print(f"  chlorophyll: carried forward on {filled} day(s), "
              f"max age {np.nanmax(age_days[age_days <= limit]):.0f}d")
    if too_old.sum():
        print(f"  chlorophyll: {int(too_old.sum())} day(s) left NaN "
              f"(older than {limit}d or before first observation)")
    return ds


def add_cyclic_month(df: pd.DataFrame) -> pd.DataFrame:
    """Encode month as sin/cos so December is adjacent to January.

    Somali waters swing between SW (Jun-Sep) and NE (Dec-Mar) monsoons, and the
    upwelling that drives productivity follows that cycle. A plain 1-12 integer
    would tell the model that month 12 and month 1 are maximally far apart.
    """
    angle = 2 * np.pi * (df["month"] - 1) / 12.0
    df["month_sin"] = np.sin(angle).astype("float32")
    df["month_cos"] = np.cos(angle).astype("float32")
    return df


# --------------------------------------------------------------- assembly

def load_sources(mode: str, year: int | None) -> dict[str, xr.Dataset]:
    """Open every downloaded source for a mode, optionally filtered to a year."""
    raw = config.region_dir(config.RAW_DIR) / mode
    if not raw.exists():
        sys.exit(f"No data at {raw}. Run: python src/download_data.py --mode {mode}"
                 + (f" --region {config.REGION}" if config.REGION != "somali" else ""))

    suffix = f"*{year}*" if year else "*"
    sources: dict[str, xr.Dataset] = {}
    for sub in sorted(p for p in raw.iterdir() if p.is_dir()):
        ds = _open_many(str(sub / f"{sub.name}_{suffix}.nc"))
        if ds is not None:
            sources[sub.name] = ds
    if not sources:
        sys.exit(f"No NetCDF files found under {raw}"
                 + (f" for year {year}" if year else ""))

    # A missing source must never pass silently. A download can fail for one
    # year (phy 2021 died with a MemoryError once) and leave a hole; building
    # that year anyway produces a table missing whole features, which then
    # trains a model on inconsistent inputs.
    expected = set(config.DATASETS_HISTORICAL if mode == "historical"
                   else config.DATASETS_NRT)
    if config.REGION != "somali":
        # Non-serving regions are training-only, so waves and wind are never
        # downloaded for them - they are excluded from the model by design.
        expected &= set(config.TRAINING_DATASETS)
    absent = expected - set(sources)
    if absent:
        print(f"  [MISSING SOURCES] {sorted(absent)} for "
              f"{year if year else 'this window'}.")
        print(f"      Re-download with: python src/download_data.py "
              f"--mode {mode}" + (f" --year {year}" if year else "")
              + f" --dataset {sorted(absent)[0]}")
    return sources


def build_year(mode: str, year: int | None, lats, lons,
               labels: pd.DataFrame | None = None,
               neg_per_pos: int | None = None) -> pd.DataFrame:
    """Build the cell x day feature table for one year.

    If `labels` is supplied the GFW join (and optional negative subsampling)
    happens per time chunk, keeping peak memory proportional to the chunk
    rather than the year.
    """
    sources = load_sources(mode, year)
    print(f"  sources: {', '.join(sorted(sources))}")

    frames: dict[str, xr.DataArray] = {}

    # ---- sea surface temperature (observed, kelvin -> celsius)
    if "sst" in sources:
        sst = _to_grid(_daily(sources["sst"]), lats, lons)["analysed_sst"] - 273.15
        frames["sst"] = sst
        frames["sst_gradient"] = sst_gradient(sst)
    # ---- model temperature: fallback for sst, and the only option in nrt mode
    if "thetao" in sources or "phy" in sources:
        key = "thetao" if "thetao" in sources else "phy"
        thetao = _to_grid(_daily(sources[key]), lats, lons)["thetao"]
        if "sst" not in frames:
            frames["sst"] = thetao
            frames["sst_gradient"] = sst_gradient(thetao)
        else:
            frames["sst_model"] = thetao

    # ---- chlorophyll (plankton; the base of the food chain)
    if "chl" in sources:
        chl = _to_grid(_daily(sources["chl"]), lats, lons)["CHL"]
        frames["chl"] = chl
        # Fish follow plankton with a delay -- trailing means capture that.
        frames["chl_lag7"] = chl.rolling(time=7, min_periods=3).mean()
        frames["chl_lag14"] = chl.rolling(time=14, min_periods=5).mean()

    # ---- currents
    cur_key = "cur" if "cur" in sources else ("phy" if "phy" in sources else None)
    if cur_key:
        cur = _to_grid(_daily(sources[cur_key]), lats, lons)
        frames["current_speed"] = np.sqrt(cur["uo"] ** 2 + cur["vo"] ** 2)

    # ---- waves (already reduced to daily mean/max at download time)
    if "wav" in sources:
        wav = _to_grid(_daily(sources["wav"]), lats, lons)
        # Wave models mask shallow water; fill the coastal strip conservatively
        # so the safety layer works where artisanal boats actually fish.
        mean_f, _ = fill_from_nearest(wav["VHM0_mean"], config.WAVE_FILL_MAX_KM)
        max_f, est = fill_from_nearest(wav["VHM0_max"], config.WAVE_FILL_MAX_KM)
        frames["wave_height"] = mean_f
        frames["wave_height_max"] = max_f
        frames["wave_estimated"] = est

    # ---- wind: components -> speed. Max matters for safety, not just mean.
    if "wnd" in sources:
        wnd = _to_grid(_daily(sources["wnd"]), lats, lons)
        frames["wind_speed"] = np.sqrt(wnd["eastward_wind_mean"] ** 2
                                       + wnd["northward_wind_mean"] ** 2)
        frames["wind_speed_max"] = np.sqrt(wnd["eastward_wind_max"] ** 2
                                           + wnd["northward_wind_max"] ** 2)

    # ---- align all variables onto one time axis
    merged = xr.Dataset(frames)
    merged = merged.dropna("time", how="all")
    if merged.sizes.get("time", 0) == 0:
        sys.exit("No overlapping days across sources - nothing to build.")
    merged = carry_chlorophyll_forward(merged)

    # Convert to rows in time slices, discarding land as we go. Materialising
    # a whole year at once is 6.3M rows for the Somali grid and 20M for the
    # wide one; the latter does not fit in memory. Land is roughly half the
    # grid, so dropping it inside the loop keeps the peak far lower than the
    # final table.
    n_time = merged.sizes["time"]
    step = max(1, min(n_time, 15))
    parts, dropped_total = [], 0
    rng = np.random.default_rng(42)
    for start in range(0, n_time, step):
        chunk = merged.isel(time=slice(start, start + step)).load()
        piece = chunk.to_dataframe().reset_index()
        chunk.close()
        piece = piece.rename(columns={"latitude": "lat", "longitude": "lon",
                                      "time": "date"})
        before = len(piece)
        piece = piece[piece["sst"].notna()]
        dropped_total += before - len(piece)
        if not len(piece):
            continue
        piece = _finalise_rows(piece)
        # Label and thin inside the loop, not after. Accumulating every ocean
        # row first is what exhausts memory on the wide grid.
        if labels is not None:
            piece = join_labels(piece, labels)
            if neg_per_pos:
                piece = subsample_negatives(piece, neg_per_pos, rng)
        parts.append(piece)
    for ds in sources.values():
        ds.close()

    if not parts:
        sys.exit("Every row was land or missing SST - nothing to build.")
    df = pd.concat(parts, ignore_index=True)
    del parts
    print(f"  dropped {dropped_total:,} land/no-data rows, kept {len(df):,}")

    lead = ["cell_id", "date", "lat", "lon", "lat_idx", "lon_idx"]
    return df[lead + [c for c in df.columns if c not in lead]]


def _finalise_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Per-chunk tidy-up: dates, month encoding, units, indices, dtypes."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["month"] = df["date"].dt.month.astype("int8")
    df = add_cyclic_month(df)

    # Wind speed m/s -> km/h, because the safety thresholds are stated in km/h.
    for col in ("wind_speed", "wind_speed_max"):
        if col in df:
            df[col] = (df[col] * 3.6).astype("float32")

    # Integer grid indices first, computed from the full-precision floats.
    lat_idx, lon_idx = cell_index(df["lat"], df["lon"])
    df["lat_idx"] = lat_idx
    df["lon_idx"] = lon_idx
    df["cell_id"] = format_cell_id(lat_idx, lon_idx).values

    # Only now downcast. lat/lon stay float32 for size; joins use the indices.
    for col in df.columns:
        if df[col].dtype == "float64":
            df[col] = df[col].astype("float32")
    return df


# ------------------------------------------------------------------ labels

def load_gfw_cells(year: int | None) -> pd.DataFrame | None:
    """Load GFW effort for a year, aggregated to grid cell-days.

    Returned separately from the join so the caller can apply it chunk by
    chunk, which is what makes the 20-million-row wide region buildable.
    """
    import download_gfw

    gfw_dir = config.region_dir(config.RAW_DIR) / "gfw"
    pattern = f"gfw_{year}-*.parquet" if year else "gfw_*.parquet"
    files = sorted(gfw_dir.glob(pattern)) if gfw_dir.exists() else []
    if not files:
        print(f"  [warn] no GFW files matching {pattern} in {gfw_dir} - "
              f"no labels.\n         Run: python src/download_gfw.py"
              + (f" --region {config.REGION}" if config.REGION != "somali" else ""))
        return None

    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cells = download_gfw.to_grid_cells(raw)
    cells["date"] = pd.to_datetime(cells["date"]).dt.normalize()
    lat_idx, lon_idx = cell_index(cells["lat"], cells["lon"])
    cells["lat_idx"] = lat_idx
    cells["lon_idx"] = lon_idx

    lats, lons = build_grid()
    inside = ((cells.lat_idx >= 0) & (cells.lat_idx < len(lats))
              & (cells.lon_idx >= 0) & (cells.lon_idx < len(lons)))
    outside = int((~inside).sum())
    cells = cells[inside]
    cells = (cells.groupby(["date", "lat_idx", "lon_idx"], as_index=False)
                  .agg(fishing_hours=("fishing_hours", "sum"),
                       n_vessels=("n_vessels", "sum")))
    print(f"  GFW: {len(cells):,} cell-days from {len(files)} month(s)"
          + (f", {outside:,} outside the grid" if outside else ""))
    return cells


def join_labels(df: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Attach fishing labels to one chunk of rows. Joins on integer indices."""
    out = df.merge(cells, on=["date", "lat_idx", "lon_idx"], how="left")
    out["fishing_hours"] = out["fishing_hours"].fillna(0.0).astype("float32")
    out["n_vessels"] = out["n_vessels"].fillna(0).astype("int16")
    out["fished"] = (out["fishing_hours"]
                     > config.FISHING_HOURS_THRESHOLD).astype("int8")
    return out


def subsample_negatives(df: pd.DataFrame, neg_per_pos: int,
                        rng: np.random.Generator) -> pd.DataFrame:
    """Keep every positive, at most neg_per_pos negatives for each."""
    pos = df[df["fished"] == 1]
    neg = df[df["fished"] == 0]
    take = min(len(neg), max(len(pos), 1) * neg_per_pos)
    if take < len(neg):
        pick = np.sort(rng.choice(len(neg), size=take, replace=False))
        neg = neg.iloc[pick]
    return pd.concat([pos, neg], ignore_index=True)


def add_fishing_labels(df: pd.DataFrame, year: int | None) -> pd.DataFrame:
    """Attach Global Fishing Watch apparent fishing effort as training labels.

    Produces:
        fishing_hours  apparent fishing hours in that cell that day (0 if none)
        n_vessels      distinct AIS vessels contributing
        fished         binary label, hours > config.FISHING_HOURS_THRESHOLD

    On the zeros: GFW reports effort wherever it detects it, so a cell-day with
    no row genuinely had no AIS-detected fishing. That is a real negative for
    the question "did tracked vessels fish here", which is the question v1
    answers. It is NOT evidence that there were no fish, and it is not evidence
    that no Somali artisanal boat was there -- those boats carry no AIS. Keep
    that distinction in the outputs.
    """
    import download_gfw

    gfw_dir = config.region_dir(config.RAW_DIR) / "gfw"
    pattern = f"gfw_{year}-*.parquet" if year else "gfw_*.parquet"
    files = sorted(gfw_dir.glob(pattern)) if gfw_dir.exists() else []
    if not files:
        print(f"  [warn] no GFW files matching {pattern} in data/raw/gfw/ - "
              f"no labels.\n         Run: python src/download_gfw.py")
        return df

    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cells = download_gfw.to_grid_cells(raw)
    cells["date"] = pd.to_datetime(cells["date"]).dt.normalize()

    # Join on integer grid indices, never on floats (see cell_index docstring).
    lat_idx, lon_idx = cell_index(cells["lat"], cells["lon"])
    cells["lat_idx"] = lat_idx
    cells["lon_idx"] = lon_idx
    # Effort just outside the grid (GFW returns whole cells overlapping the
    # polygon) would otherwise merge onto nothing; drop it explicitly.
    inside = ((cells.lat_idx >= 0) & (cells.lat_idx < len(build_grid()[0]))
              & (cells.lon_idx >= 0) & (cells.lon_idx < len(build_grid()[1])))
    dropped = int((~inside).sum())
    cells = cells[inside]

    before = len(cells)
    cells = (cells.groupby(["date", "lat_idx", "lon_idx"], as_index=False)
                  .agg(fishing_hours=("fishing_hours", "sum"),
                       n_vessels=("n_vessels", "sum")))
    out = df.merge(cells, on=["date", "lat_idx", "lon_idx"], how="left")

    matched = int(out["fishing_hours"].notna().sum())
    print(f"  GFW: {before:,} cell-days available, {matched:,} matched the "
          f"feature grid" + (f", {dropped:,} outside the box" if dropped else ""))
    if before and matched / before < 0.5:
        print("  [warn] fewer than half the GFW cell-days matched - check the "
              "grid alignment before trusting these labels.")

    out["fishing_hours"] = out["fishing_hours"].fillna(0.0).astype("float32")
    out["n_vessels"] = out["n_vessels"].fillna(0).astype("int16")
    out["fished"] = (out["fishing_hours"] > config.FISHING_HOURS_THRESHOLD).astype("int8")

    rate = out["fished"].mean() * 100
    print(f"  labels: {int(out['fished'].sum()):,} fished cell-days "
          f"({rate:.2f}% positive) from {len(files)} GFW month(s)")
    if rate < 0.5:
        print("  [note] very few positives - expect a strongly imbalanced "
              "classifier; use PR-AUC, not accuracy.")
    return out


# ------------------------------------------------------------- bathymetry

def add_bathymetry(df: pd.DataFrame, lats, lons) -> pd.DataFrame:
    """Attach depth and distance-to-coast from GEBCO, if available.

    GEBCO is a one-time manual download (docs/DATA_SOURCES.md section 2); it is
    not fetched by download_data.py. Depth is one of the strongest predictors
    and is also one of the two required baselines, so this is not optional for
    the final model -- but the table is still useful without it.
    """
    gebco = next(iter(glob.glob(str(config.region_dir(config.RAW_DIR)
                                    / "gebco" / "*.nc"))), None)
    if gebco is None:
        print("  [warn] no GEBCO file in data/raw/gebco/ - skipping depth and "
              "dist_coast.\n         Download the bounding-box cutout: "
              "https://download.gebco.net")
        return df

    ds = xr.open_dataset(gebco)
    elev_name = "elevation" if "elevation" in ds else list(ds.data_vars)[0]
    ren = {}
    for cand in ("lat", "y"):
        if cand in ds.coords:
            ren[cand] = "latitude"
    for cand in ("lon", "x"):
        if cand in ds.coords:
            ren[cand] = "longitude"
    elev_hi = ds.rename(ren)[elev_name]

    # GEBCO is ~15 arcsec (1/240 deg); our grid is 0.1 deg, exactly 24x coarser.
    # Block-AVERAGE rather than point-sample: one interpolated point is a
    # single sounding, whereas the mean over the block is the cell's actual
    # typical depth. Fall back to interpolation if the ratio is not integral.
    ratio_lat = config.GRID_RES / float(abs(elev_hi.latitude[1] - elev_hi.latitude[0]))
    ratio_lon = config.GRID_RES / float(abs(elev_hi.longitude[1] - elev_hi.longitude[0]))
    integral = (abs(ratio_lat - round(ratio_lat)) < 1e-6
                and abs(ratio_lon - round(ratio_lon)) < 1e-6
                and elev_hi.sizes["latitude"] % round(ratio_lat) == 0
                and elev_hi.sizes["longitude"] % round(ratio_lon) == 0)

    if integral:
        blocks = elev_hi.coarsen(latitude=round(ratio_lat),
                                 longitude=round(ratio_lon))
        elev = blocks.mean()
        # Roughness: seabed variability inside the cell. Shelf breaks, banks
        # and seamounts concentrate fish, and a flat mean hides them.
        roughness = blocks.std()
        # Fraction of the cell that is dry land, for a soft coastline.
        land_frac = (elev_hi >= 0).coarsen(latitude=round(ratio_lat),
                                           longitude=round(ratio_lon)).mean()
        elev = elev.assign_coords(latitude=lats, longitude=lons)
        roughness = roughness.assign_coords(latitude=lats, longitude=lons)
        land_frac = land_frac.assign_coords(latitude=lats, longitude=lons)
        print(f"  GEBCO: block-averaged {round(ratio_lat)}x{round(ratio_lon)} "
              f"cells onto the 0.1 deg grid")
    else:
        elev = elev_hi.interp(latitude=lats, longitude=lons, method="linear")
        roughness = xr.full_like(elev, np.nan)
        land_frac = (elev >= 0).astype("float32")
        print("  GEBCO: grid not an integer multiple - interpolated instead")

    # These are static per cell, so look them up by grid index instead of
    # merging. A pandas merge onto a 6-million-row table promotes everything
    # to float64 and blows past available memory.
    planes = {
        "depth": (-elev).values.astype("float32"),
        "seabed_roughness": np.asarray(roughness.values, dtype="float32"),
        "land_fraction": np.asarray(land_frac.values, dtype="float32"),
    }
    ds.close()

    depth_plane = planes["depth"]
    above_sea = depth_plane <= 0
    if above_sea.any():
        print(f"  [note] {int(above_sea.sum()):,} grid cells sit at or above "
              f"sea level in GEBCO (coastline blocks); depth set to NaN there.")
        depth_plane[above_sea] = np.nan

    out = df
    ii = out["lat_idx"].to_numpy()
    jj = out["lon_idx"].to_numpy()
    for name, plane in planes.items():
        out[name] = plane[ii, jj]

    ocean = out["depth"] > 0
    print(f"  added depth: median {out.loc[ocean, 'depth'].median():.0f} m, "
          f"range {out.loc[ocean, 'depth'].min():.0f}-"
          f"{out.loc[ocean, 'depth'].max():.0f} m; "
          f"plus seabed_roughness, land_fraction")
    return out


# ------------------------------------------------------------------- main

def list_downloaded() -> None:
    """Report what has been downloaded so far, per mode and dataset."""
    for mode in ("historical", "nrt"):
        root = config.RAW_DIR / mode
        print(f"\n{mode}:")
        if not root.exists():
            print("  (nothing downloaded)")
            continue
        for sub in sorted(p for p in root.iterdir() if p.is_dir()):
            files = sorted(sub.glob("*.nc"))
            mb = sum(f.stat().st_size for f in files) / 1e6
            labels = [f.stem.replace(f"{sub.name}_", "") for f in files]
            span = f"{labels[0]} .. {labels[-1]}" if labels else "-"
            print(f"  {sub.name:8s} {len(files):3d} file(s)  {mb:8.1f} MB   {span}")
    gebco = list(glob.glob(str(config.RAW_DIR / "gebco" / "*.nc")))
    print(f"\ngebco: {'present' if gebco else 'MISSING (manual download)'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["historical", "nrt"], default="historical")
    parser.add_argument("--year", type=int, help="build only this year")
    parser.add_argument("--list", action="store_true",
                        help="show what has been downloaded, then exit")
    parser.add_argument("--no-bathymetry", action="store_true",
                        help="skip the GEBCO depth join")
    parser.add_argument("--no-labels", action="store_true",
                        help="skip the GFW fishing-effort label join")
    parser.add_argument("--region", choices=list(config.REGIONS), default="somali",
                        help="'wide' builds the western Indian Ocean training "
                             "region (55,000 cells) into its own files")
    parser.add_argument("--neg-per-pos", type=int, default=None,
                        help="keep at most N negative rows per positive, "
                             "dropping the rest. All positives are always "
                             "kept. Needed for the wide region, where a full "
                             "year is ~14M rows. Training data only - never "
                             "use on anything you will evaluate on.")
    args = parser.parse_args()

    config.set_region(args.region)

    if args.list:
        list_downloaded()
        return

    lats, lons = build_grid()
    print(f"Grid: {len(lats)} x {len(lons)} = {len(lats) * len(lons):,} cells "
          f"at {config.GRID_RES} deg")

    if args.mode == "nrt":
        years: list[int | None] = [None]
    elif args.year:
        years = [args.year]
    else:
        end_year = dt.date.fromisoformat(config.HISTORICAL_END).year
        years = list(range(dt.date.fromisoformat(config.START_DATE).year, end_year + 1))

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for year in years:
        label = str(year) if year else "latest"
        print(f"\n[{args.mode} {label}]")
        want_labels = args.mode == "historical" and not args.no_labels
        labels = load_gfw_cells(year) if want_labels else None
        try:
            df = build_year(args.mode, year, lats, lons, labels,
                            args.neg_per_pos)
        except SystemExit as exc:
            print(f"  skipped: {exc}")
            continue
        if not args.no_bathymetry:
            df = add_bathymetry(df, lats, lons)

        if "fished" in df:
            rate = df["fished"].mean() * 100
            print(f"  labels: {int(df['fished'].sum()):,} fished cell-days "
                  f"({rate:.2f}% positive"
                  + (f", negatives thinned {args.neg_per_pos}:1"
                     if args.neg_per_pos else "") + ")")

        prefix = f"features_{args.mode}"
        if args.region != "somali":
            prefix = f"features_{args.region}_{args.mode}"
        out = config.PROCESSED_DIR / f"{prefix}_{label}.parquet"
        df.to_parquet(out, index=False)
        written.append(out)
        print(f"  wrote {out.name}  ({len(df):,} rows x {len(df.columns)} cols, "
              f"{out.stat().st_size / 1e6:.1f} MB)")
        print(f"  days: {df['date'].min().date()} -> {df['date'].max().date()}")

    if not written:
        sys.exit("\nNothing built. Check that downloads have completed.")
    print(f"\nBuilt {len(written)} file(s) in {config.PROCESSED_DIR}")
    print("Next: fishing-effort labels from Global Fishing Watch, "
          "then src/train_model.py")


if __name__ == "__main__":
    main()
