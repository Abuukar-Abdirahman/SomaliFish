"""Central configuration for SomaliFish AI. All constants live here."""

from pathlib import Path

# ---------------------------------------------------------------- study area
# Somali EEZ bounding box (generous, includes offshore waters)
LAT_MIN = -2.0
LAT_MAX = 12.5
LON_MIN = 40.5
LON_MAX = 52.5

# Grid resolution in degrees (~11 km at the equator)
GRID_RES = 0.1

# ------------------------------------------------------------------ regions
# The Somali box alone yielded a model that could not beat a climatology
# baseline: too few distinct fishing situations to learn environment ->
# fishing from. The wider western Indian Ocean carries ~18x the fishing
# effort (35,653 vessel-day rows/month vs 1,939), so we can train there and
# predict on Somali waters. Serving and the map always use "somali".
REGIONS = {
    "somali": {"lat": (-2.0, 12.5), "lon": (40.5, 52.5),
               "note": "Somali EEZ - the area we serve"},
    "wide":   {"lat": (-10.0, 15.0), "lon": (38.0, 60.0),
               "note": "western Indian Ocean - training only, ~3.2x the area"},
}
REGION = "somali"


def set_region(name: str) -> None:
    """Switch the active bounding box. Call before anything reads LAT_MIN.

    Modules read config.LAT_MIN at call time rather than import time, so this
    reconfigures the whole pipeline. Raw data is stored under a per-region
    directory so the two never mix.
    """
    global LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, REGION
    if name not in REGIONS:
        raise ValueError(f"unknown region {name!r}; choose from {list(REGIONS)}")
    LAT_MIN, LAT_MAX = REGIONS[name]["lat"]
    LON_MIN, LON_MAX = REGIONS[name]["lon"]
    REGION = name


def region_dir(base):
    """Per-region subdirectory, so Somali and wide data never mix."""
    return base if REGION == "somali" else base / f"region_{REGION}"


# Datasets needed for TRAINING only. Waves and wind are excluded from the
# hotspot model on purpose (they predict weather, not fish - see
# train_model.py), so the wider region does not need them. That also skips
# the two sub-daily products, which are by far the slowest to download.
TRAINING_DATASETS = ["sst", "chl", "phy"]

# ---------------------------------------------------------------- time range
START_DATE = "2019-01-01"   # v1 training window start
END_DATE = None             # None = today

TRAIN_END_YEAR = 2023       # temporal holdout: train <= this, test after

# The historical (reanalysis/reprocessed) products lag real time. The training
# table can only span where ALL historical features exist; the binding
# constraint is reprocessed SST (ends 2026-03-31), then wind (2026-04-20).
# Keep this in sync if the products advance.
HISTORICAL_END = "2026-03-31"

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PREDICTIONS_DIR = DATA_DIR / "predictions"
DEMO_DIR = DATA_DIR / "demo"
MODELS_DIR = ROOT / "models"

for _d in (RAW_DIR, PROCESSED_DIR, PREDICTIONS_DIR, DEMO_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------- Copernicus Marine datasets
# Two families, because no single Copernicus product covers both a long
# training history AND today:
#
#   HISTORICAL ("my" / "REP" = multi-year reanalysis & reprocessed observations)
#       Long records (1980s/90s onwards) but they lag real time by 2-5 months.
#       -> use these for TRAINING (2019 -> 2023 train, 2024+ test).
#
#   NRT ("anfc" / "NRT" = analysis-forecast & near-real-time)
#       Updated to today (and a few days ahead) but short histories, several
#       only starting 2022-2024.
#       -> use these for the DAILY PREDICTION / live demo.
#
# Verified against the catalogue on 2026-08-11 (coverage noted per entry).
# Re-verify with:  python src/download_data.py --check
# If an ID stops resolving, search https://data.marine.copernicus.eu using the
# product names in docs/DATA_SOURCES.md.
#
# Fields: dataset_id, variables, cadence (native time step), coverage note.

DATASETS_HISTORICAL = {
    "sst": {
        "dataset_id": "METOFFICE-GLO-SST-L4-REP-OBS-SST",
        "variables": ["analysed_sst"],          # kelvin
        "cadence": "P1D",
        "coverage": "1981-10-01 -> 2026-03-31",
    },
    "chl": {
        "dataset_id": "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D",
        "variables": ["CHL"],                   # mg/m3, gap-filled L4
        "cadence": "P1D",
        "coverage": "1997-09-04 -> 2026-08-03",
    },
    "phy": {
        "dataset_id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "variables": ["uo", "vo", "thetao"],    # currents + model SST
        "cadence": "P1D",
        "coverage": "1993-01-01 -> 2026-06-23",
        "surface_only": True,   # 50 depth levels otherwise; we only fish the top
    },
    "wav": {
        "dataset_id": "cmems_mod_glo_wav_my_0.2deg_PT3H-i",
        "variables": ["VHM0"],                  # significant wave height, m
        "cadence": "PT3H",
        "coverage": "1980-01-01 -> 2026-05-31",
    },
    "wnd": {
        "dataset_id": "cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H",
        "variables": ["eastward_wind", "northward_wind"],   # m/s
        "cadence": "PT1H",
        "coverage": "2007-01-11 -> 2026-04-20",
    },
}

DATASETS_NRT = {
    "sst": {
        "dataset_id": "METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2",
        "variables": ["analysed_sst"],
        "cadence": "P1D",
        "coverage": "2024-01-17 -> today",
    },
    "chl": {
        # same multi-year gap-free product: it tracks to ~1 week behind today,
        # which is close enough for the daily map and avoids a sensor change.
        "dataset_id": "cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D",
        "variables": ["CHL"],
        "cadence": "P1D",
        "coverage": "1997-09-04 -> today minus ~7d",
    },
    "cur": {
        "dataset_id": "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m",
        "variables": ["uo", "vo"],
        "cadence": "P1D",
        "coverage": "2022-06-01 -> today +9d",
        "surface_only": True,
    },
    "thetao": {
        "dataset_id": "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m",
        "variables": ["thetao"],
        "cadence": "P1D",
        "coverage": "2022-06-01 -> today +9d",
        "surface_only": True,
    },
    "wav": {
        "dataset_id": "cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
        "variables": ["VHM0"],
        "cadence": "PT3H",
        "coverage": "2022-11-01 -> today +9d",
    },
    "wnd": {
        "dataset_id": "cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
        "variables": ["eastward_wind", "northward_wind"],
        "cadence": "PT1H",
        "coverage": "2024-06-13 -> today",
    },
}

# Sub-daily datasets are reduced to daily statistics right after download,
# otherwise hourly wind alone is ~800 MB/year over our box.
SUBDAILY = {"wav", "wnd"}

# Chlorophyll publishes ~7-10 days behind real time, so "today" never has a
# value. We fetch a longer tail and carry the most recent observation forward,
# recording how old it is in the `chl_age_days` feature. This is defensible
# rather than a fudge: the model already uses 7- and 14-day trailing means,
# because fish respond to plankton with a lag, not to today's exact reading.
NRT_CHL_LOOKBACK_DAYS = 40
CHL_MAX_CARRY_FORWARD_DAYS = 21   # beyond this, leave NaN rather than pretend

# ---------------------------------------------------------------- species v1
TARGET_SPECIES = {
    "yellowfin_tuna": "Thunnus albacares",
    "skipjack_tuna": "Katsuwonus pelamis",
    "kingfish": "Scomberomorus commerson",
    "snapper": "Lutjanus",
    "sardine": "Sardinella",
    "grouper": "Epinephelus",
}

# Wider box for species occurrence harvesting (more signal)
SPECIES_LAT_MIN, SPECIES_LAT_MAX = -10.0, 15.0
SPECIES_LON_MIN, SPECIES_LON_MAX = 38.0, 60.0

# ------------------------------------------------------------- landing sites
# Somali fishing ports and landing sites, Somali name first. Coordinates are
# the harbour/beach, so distances are "from where the boat launches".
# Verify and extend with fishermen (docs/ROADMAP.md Phase 1) - this list is
# from public geography, not from field work.
LANDING_SITES = [
    {"id": "boosaaso",  "so": "Boosaaso",  "en": "Bosaso",    "lat": 11.284, "lon": 49.182},
    {"id": "caluula",   "so": "Caluula",   "en": "Alula",     "lat": 11.966, "lon": 50.757},
    {"id": "qandala",   "so": "Qandala",   "en": "Qandala",   "lat": 11.472, "lon": 49.874},
    {"id": "berbera",   "so": "Berbera",   "en": "Berbera",   "lat": 10.435, "lon": 45.016},
    {"id": "eyl",       "so": "Eyl",       "en": "Eyl",       "lat":  7.981, "lon": 49.816},
    {"id": "garacad",   "so": "Garacad",   "en": "Garacad",   "lat":  6.964, "lon": 49.371},
    {"id": "hobyo",     "so": "Hobyo",     "en": "Hobyo",     "lat":  5.351, "lon": 48.527},
    {"id": "cadale",    "so": "Cadale",    "en": "Adale",     "lat":  2.747, "lon": 46.309},
    {"id": "muqdisho",  "so": "Muqdisho",  "en": "Mogadishu", "lat":  2.037, "lon": 45.343},
    {"id": "marka",     "so": "Marka",     "en": "Merca",     "lat":  1.716, "lon": 44.771},
    {"id": "baraawe",   "so": "Baraawe",   "en": "Brava",     "lat":  1.112, "lon": 44.028},
    {"id": "kismaayo",  "so": "Kismaayo",  "en": "Kismayo",   "lat": -0.358, "lon": 42.545},
]

# How far a small Somali fishing boat can realistically work in a day trip.
# An open boat with an outboard, no radar, returning before dark. A hotspot
# beyond this is not advice, it is a hazard - so the map must mark the limit.
# Provisional: confirm with fishermen before trusting these numbers.
ARTISANAL_MAX_COAST_KM = 50.0    # distance offshore
ARTISANAL_MAX_TRIP_KM = 80.0     # distance from the launching site
ARTISANAL_BOAT_SPEED_KMH = 18.0  # typical loaded outboard skiff, for travel time

# Distance-from-coast bands used to divide each site's grounds into zones.
COAST_BANDS_KM = [(0, 15, "Xeebta", "Inshore"),
                  (15, 35, "Dhexe", "Mid-shore"),
                  (35, 80, "Dibadda", "Offshore")]

# ---------------------------------------------------------------- labels v1
# A cell-day counts as "fished" if apparent fishing hours exceed this. Above
# zero because AIS position noise smears a little effort into neighbouring
# cells; a vessel genuinely working a cell logs hours, not minutes.
FISHING_HOURS_THRESHOLD = 0.5

# Months when the AIS fleet leaves Somali waters for the SW monsoon, measured
# from unsubsampled 2019 data: July saw 62 vessel-days across the whole EEZ
# against January's 5,316. Absence in these months says nothing about fish,
# so they are excluded from training. Used as a fallback when the data has
# been class-balanced and dormancy can no longer be detected from prevalence.
FLEET_DORMANT_MONTHS = [7, 8]

# ---------------------------------------------------------------- safety
# Wave models mask shallow water: ~77% of cells within 15 km of the Somali
# coast have no wave height, which is precisely where artisanal boats work.
# Those gaps are filled from the nearest valid cell up to this distance and
# flagged as estimated. Filling from offshore over-warns rather than
# under-warns, which is the safe direction for a safety system.
WAVE_FILL_MAX_KM = 30.0

WAVE_SAFE_M = 1.25
WAVE_DANGER_M = 2.0
WIND_SAFE_KMH = 20.0
WIND_DANGER_KMH = 30.0
