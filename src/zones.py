"""Turn the 0.1 degree grid into fishing zones a person can actually use.

A fisherman cannot act on 8,294 coloured squares. He needs a small number of
named places, ranked, with a distance and a travel time from where his boat
launches. This module builds those.

Three ideas do the work:

1. REACHABILITY. The grid extends ~700 km offshore. A Somali artisanal boat -
   open skiff, outboard, back before dark - works within roughly 50 km of the
   coast. A hotspot beyond that range is not advice, it is a hazard. Cells are
   tagged reachable / not reachable and the map marks the limit.

2. ZONES. Each reachable cell is assigned to its nearest landing site and to a
   distance band (inshore / mid / offshore). Site x band = one named zone,
   e.g. "Xeebta Muqdisho". That is ~30 zones instead of 8,294 cells.

3. RANKING. Zones are ranked by a fishing-potential index. Two sources, in
   order of preference:
     - the trained hotspot model, once it exists;
     - otherwise a transparent PHYSICS INDEX from measured chlorophyll and
       temperature fronts (see physics_index).
   Which one was used is always reported, and the physics index is never
   presented as a model prediction.

Distance to coast is derived from the land mask in the satellite data itself
(sea surface temperature is absent over land), so this needs no GEBCO file.
GEBCO is still required for depth.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

EARTH_R_KM = 6371.0


# ------------------------------------------------------------ distance tools

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km. Accepts scalars or arrays."""
    lat1, lon1, lat2, lon2 = map(np.radians, (np.asarray(lat1, dtype="float64"),
                                              np.asarray(lon1, dtype="float64"),
                                              np.asarray(lat2, dtype="float64"),
                                              np.asarray(lon2, dtype="float64")))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def distance_to_coast(df: pd.DataFrame, lats: np.ndarray,
                      lons: np.ndarray) -> pd.Series:
    """Kilometres from each ocean cell to the nearest land cell.

    The land mask comes free with the data: satellite SST is undefined over
    land, so any grid cell absent from the ocean feature table is land (or
    outside the satellite's domain). No bathymetry file needed.
    """
    from scipy.ndimage import distance_transform_edt

    ocean = np.zeros((len(lats), len(lons)), dtype=bool)
    ocean[df["lat_idx"].to_numpy(), df["lon_idx"].to_numpy()] = True
    land = ~ocean

    if not land.any():
        return pd.Series(np.nan, index=df.index)

    # Anisotropic spacing: a degree of longitude is shorter than a degree of
    # latitude away from the equator.
    dy = config.GRID_RES * 111.32
    dx = config.GRID_RES * 111.32 * float(np.cos(np.deg2rad(np.mean(lats))))
    dist = distance_transform_edt(ocean, sampling=(dy, dx))
    return pd.Series(dist[df["lat_idx"].to_numpy(), df["lon_idx"].to_numpy()],
                     index=df.index).astype("float32")


def nearest_landing_site(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each cell to its closest landing site, with the distance."""
    sites = config.LANDING_SITES
    lat = df["lat"].to_numpy(dtype="float64")
    lon = df["lon"].to_numpy(dtype="float64")

    dists = np.empty((len(sites), len(df)), dtype="float32")
    for i, site in enumerate(sites):
        dists[i] = haversine_km(lat, lon, site["lat"], site["lon"])

    best = np.argmin(dists, axis=0)
    out = pd.DataFrame(index=df.index)
    out["site_id"] = [sites[i]["id"] for i in best]
    out["site_km"] = dists[best, np.arange(len(df))]
    return out


# ------------------------------------------------------------ physics index

def physics_index(df: pd.DataFrame) -> pd.Series:
    """Transparent fishing-potential indicator from measured ocean conditions.

    THIS IS NOT A MODEL PREDICTION. It is a weighted blend of two measured
    quantities that fisheries science links to productive water, each converted
    to a percentile rank across the grid for the day:

      * chlorophyll-a - plankton, the base of the food chain (weight 0.55)
      * sea-surface temperature gradient - fronts, where prey concentrates,
        the classic Potential Fishing Zone signal used by India's INCOIS
        advisories (weight 0.45)

    The weights are a reasoned starting point, NOT calibrated against Somali
    catch data - that data does not exist yet. Treat the output as "where the
    water looks biologically promising today", relative to the rest of the
    grid, and label it that way everywhere it is shown.

    Returns 0-100, or NaN where inputs are missing.
    """
    parts, weights = [], []
    if "chl" in df:
        # Chlorophyll spans orders of magnitude; rank is more robust than
        # the raw value and needs no threshold tuning.
        parts.append(df["chl"].rank(pct=True) * 100)
        weights.append(0.55)
    if "sst_gradient" in df:
        parts.append(df["sst_gradient"].rank(pct=True) * 100)
        weights.append(0.45)

    if not parts:
        return pd.Series(np.nan, index=df.index)

    stacked = pd.concat(parts, axis=1)
    w = np.array(weights, dtype="float64")
    valid = stacked.notna()
    weighted = (stacked.fillna(0) * w).sum(axis=1)
    total_w = (valid * w).sum(axis=1)
    score = np.where(total_w > 0, weighted / np.maximum(total_w, 1e-9), np.nan)
    return pd.Series(score, index=df.index).round(1)


# ------------------------------------------------------------------- zoning

def band_for(dist_km: float) -> tuple[str, str, str] | None:
    for lo, hi, so, en in config.COAST_BANDS_KM:
        if lo <= dist_km < hi:
            return f"{lo}-{hi}", so, en
    return None


def assign(df: pd.DataFrame, lats: np.ndarray, lons: np.ndarray) -> pd.DataFrame:
    """Add reachability, landing site, distance band and zone id to each cell."""
    out = df.copy()
    out["dist_coast_km"] = distance_to_coast(out, lats, lons)
    sites = nearest_landing_site(out)
    out["site_id"] = sites["site_id"]
    out["site_km"] = sites["site_km"]

    out["reachable"] = ((out["dist_coast_km"] <= config.ARTISANAL_MAX_COAST_KM)
                        & (out["site_km"] <= config.ARTISANAL_MAX_TRIP_KM))

    bands = out["dist_coast_km"].apply(
        lambda d: band_for(d) if pd.notna(d) else None)
    out["band"] = [b[0] if b else None for b in bands]
    out["band_so"] = [b[1] if b else None for b in bands]
    out["zone_id"] = np.where(out["band"].notna(),
                              out["site_id"] + "_" + out["band"].astype(str),
                              None)
    return out


def summarise(df: pd.DataFrame, score_col: str, score_source: str,
              top_n: int | None = None) -> list[dict]:
    """Aggregate cells into ranked zones.

    Only reachable cells form zones: ranking water a small boat cannot get to
    would be worse than showing nothing.
    """
    import safety

    usable = df[df["reachable"] & df["zone_id"].notna()].copy()
    if usable.empty:
        return []

    site_by_id = {s["id"]: s for s in config.LANDING_SITES}
    band_names = {f"{lo}-{hi}": (so, en) for lo, hi, so, en in config.COAST_BANDS_KM}

    zones = []
    for zone_id, grp in usable.groupby("zone_id"):
        site = site_by_id[grp["site_id"].iloc[0]]
        band_so, band_en = band_names[grp["band"].iloc[0]]

        status = safety.classify(grp.get("wave_height_max"),
                                 grp.get("wind_speed_max"))
        counts = status.value_counts()
        # The zone's safety is its WORST common condition, not its average.
        share_danger = counts.get(safety.DANGER, 0) / len(grp)
        if share_danger >= 0.34:
            zone_status = safety.DANGER
        elif counts.get(safety.SAFE, 0) / len(grp) >= 0.5:
            zone_status = safety.SAFE
        elif counts.get(safety.UNKNOWN, 0) / len(grp) > 0.5:
            zone_status = safety.UNKNOWN
        else:
            zone_status = safety.CAUTION

        score = grp[score_col].mean() if score_col in grp else np.nan
        km = float(grp["site_km"].mean())

        zones.append({
            "zone_id": zone_id,
            "name_so": f"{band_so} {site['so']}",
            "name_en": f"{site['en']} {band_en.lower()}",
            "site": {"id": site["id"], "so": site["so"], "en": site["en"],
                     "lat": site["lat"], "lon": site["lon"]},
            "band_km": grp["band"].iloc[0],
            "n_cells": int(len(grp)),
            "centre": {"lat": round(float(grp["lat"].mean()), 3),
                       "lon": round(float(grp["lon"].mean()), 3)},
            # Extent, so the map can outline the zone as a crisp vector
            # instead of relying on the interpolated raster.
            "bounds": {
                "lat_min": round(float(grp["lat"].min()) - config.GRID_RES / 2, 3),
                "lat_max": round(float(grp["lat"].max()) + config.GRID_RES / 2, 3),
                "lon_min": round(float(grp["lon"].min()) - config.GRID_RES / 2, 3),
                "lon_max": round(float(grp["lon"].max()) + config.GRID_RES / 2, 3),
            },
            "distance_km": round(km, 1),
            "travel_hours": round(km / config.ARTISANAL_BOAT_SPEED_KMH, 1),
            "score": None if pd.isna(score) else round(float(score), 1),
            "score_source": score_source,
            "safety": zone_status,
            "share_dangerous": round(float(share_danger), 2),
            "conditions": {
                "sst": _mean(grp, "sst", 1),
                "chl": _mean(grp, "chl", 3),
                "sst_gradient": _mean(grp, "sst_gradient", 2),
                "wave_height_max": _mean(grp, "wave_height_max", 2),
                "wind_speed_max": _mean(grp, "wind_speed_max", 0),
            },
            # Share of the zone whose wave height was filled from a
            # neighbouring cell rather than measured there.
            "wave_estimated_share": (
                round(float(grp["wave_estimated"].mean()), 2)
                if "wave_estimated" in grp else 0.0),
        })

    # Rank on score, but never rank a dangerous zone above a safe one: the
    # ordering is advice, and advice must not send someone into a storm.
    danger_rank = {safety.SAFE: 0, safety.CAUTION: 1,
                   safety.UNKNOWN: 2, safety.DANGER: 3}
    zones.sort(key=lambda z: (danger_rank.get(z["safety"], 9),
                              -(z["score"] if z["score"] is not None else -1)))
    for i, z in enumerate(zones, 1):
        z["rank"] = i
    return zones[:top_n] if top_n else zones


def _mean(grp: pd.DataFrame, col: str, decimals: int):
    if col not in grp:
        return None
    val = grp[col].mean()
    return None if pd.isna(val) else round(float(val), decimals)


# Somali-language rating for a zone score, so the map never shows a bare
# number to someone who wants a plain answer.
def rating(score: float | None) -> dict:
    if score is None:
        return {"so": "Lama oga", "en": "Unknown", "level": "unknown"}
    if score >= 70:
        return {"so": "Rajo fiican", "en": "Promising", "level": "high"}
    if score >= 45:
        return {"so": "Dhexdhexaad", "en": "Moderate", "level": "mid"}
    return {"so": "Rajo yar", "en": "Poor", "level": "low"}
