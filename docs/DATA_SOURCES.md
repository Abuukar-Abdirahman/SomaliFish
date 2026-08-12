# Data sources

Every dataset used by SomaliFish AI, with exact access method and variables.
All sources are free. Study area: lat -2.0 → 12.5, lon 40.5 → 52.5 (Somali EEZ
bounding box, see `src/config.py`).

## 1. Copernicus Marine — ocean physics & color

Account: https://data.marine.copernicus.eu (free registration).
Access: official `copernicusmarine` Python package (`pip install copernicusmarine`,
then `copernicusmarine login` once).

### Why there are two sets of datasets

No single Copernicus product covers both a long training history *and* today:

- **Reanalysis / reprocessed** products (`_my_`, `REP`) go back to the 1980s–90s
  but lag real time by 2–5 months. → **training**.
- **Analysis-forecast / near-real-time** products (`anfc`, `NRT`) run to today
  and a few days ahead, but several only begin in 2022–2024. → **daily
  prediction / live demo**.

Using only the NRT products (the original plan) would have left *zero* training
data before 2024, making the train ≤2023 / test 2024+ holdout impossible.

### Historical (training) — `config.DATASETS_HISTORICAL`

| What | Dataset ID | Variables | Coverage |
|---|---|---|---|
| SST | `METOFFICE-GLO-SST-L4-REP-OBS-SST` | analysed_sst (K) | 1981-10 → 2026-03-31 |
| Chlorophyll-a | `cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D` | CHL (mg/m³) | 1997-09 → 2026-08-03 |
| Currents + temp | `cmems_mod_glo_phy_my_0.083deg_P1D-m` | uo, vo, thetao | 1993-01 → 2026-06-23 |
| Waves | `cmems_mod_glo_wav_my_0.2deg_PT3H-i` | VHM0 (m), 3-hourly | 1980-01 → 2026-05-31 |
| Wind | `cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H` | eastward/northward_wind (m/s), hourly | 2007-01 → 2026-04-20 |

Binding constraint on the training window: reprocessed SST ends **2026-03-31**
(`config.HISTORICAL_END`). Training window for v1: 2019-01-01 → 2026-03-31,
which gives 5 years of training (≤2023) and 2+ years of held-out test (2024+).

### Near-real-time (daily prediction) — `config.DATASETS_NRT`

| What | Dataset ID | Variables |
|---|---|---|
| SST | `METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2` | analysed_sst |
| Chlorophyll-a | (same gap-free product as above, ~7 days behind) | CHL |
| Currents | `cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m` | uo, vo |
| Temperature | `cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m` | thetao |
| Waves | `cmems_mod_glo_wav_anfc_0.083deg_PT3H-i` | VHM0 |
| Wind | `cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H` | eastward/northward_wind |

Note the `anfc` products publish ~9 days *ahead* of today, so the map can show
a genuine short-range forecast, not just the latest observation.

**All IDs above verified against the live catalogue on 2026-08-11.** IDs change
between product versions; re-verify any time (no account needed) with:

```bash
python src/download_data.py --check              # historical
python src/download_data.py --check --mode nrt   # near-real-time
```

Notes:
- Chlorophyll has cloud gaps → the L4 **gap-free** product above is used.
- Download as NetCDF, read with xarray.
- Keep requests small: one dataset, one year, our bounding box per request.
- Waves (3-hourly) and wind (hourly) are downloaded **month by month** and
  reduced to daily mean + daily max immediately, before the next chunk. Raw
  hourly wind over our box is ~800 MB/year; the reduction keeps the whole
  historical archive to a few GB. Daily **max** matters independently of the
  mean — it is what the safety module should threshold on.

## 2. GEBCO — bathymetry (depth)

- https://www.gebco.net → GEBCO 2024 Grid, global NetCDF (~7 GB global, or use
  the web download tool to cut only our bounding box, much smaller).
- One-time download. Variable: elevation (negative = depth below sea level).
- Derived features: depth at cell, distance to coast, slope (optional).

## 3. Global Fishing Watch — fishing activity (v1 labels)

- https://globalfishingwatch.org → free account → API access token.
- API docs: https://globalfishingwatch.org/our-apis/
- Use the "4Wings" fishing-effort endpoint: apparent fishing hours,
  gridded (0.1° available), filterable by date range and region.
- Request fishing effort for our bounding box, 2019 → present, monthly or daily.

KNOWN LIMITATION (document everywhere): GFW is based mainly on AIS transponders
carried by industrial vessels. Somali artisanal boats mostly have no AIS, so
GFW shows industrial + foreign activity, not small-boat reality. v1 therefore
predicts "conditions attractive for fishing activity", a proxy for productive
zones. Ground truth from Somali fishermen replaces this in v2.

## 4. OBIS / GBIF — species presence records

- OBIS: https://obis.org — marine species observations. Python: `pyobis`,
  or REST: https://api.obis.org/v3/occurrence?geometry=<WKT>&scientificname=<name>
- GBIF: https://www.gbif.org — REST API, no auth for search.
- Target species (initial, refine after fishermen interviews):
  - Yellowfin tuna — Thunnus albacares
  - Skipjack tuna — Katsuwonus pelamis
  - Narrow-barred Spanish mackerel / "kingfish" — Scomberomorus commerson
  - Snapper — Lutjanus spp.
  - Sardine — Sardinella spp.
  - Grouper — Epinephelus spp.
- Pull all occurrence records within the wider western Indian Ocean box
  (lat -10 → 15, lon 38 → 60) for more training signal; predict on our grid.
- Species model needs pseudo-absences: sample random background cells with
  environmental data as negatives (standard species-distribution-modeling
  practice, same idea as MaxEnt background points).

## 5. FAO / national statistics — context only

- FAO FishStatJ and FAO Somalia reports give annual national catch estimates
  and species composition. Not gridded, not for training — use for the pitch,
  species prioritization, and sanity checks.

## 6. Future: Somali fishermen catch logs (v2 ground truth)

Collected via our own mobile app (see ROADMAP). Schema target:
date, lat, lon (or zone), species, weight_range_kg, count, method, depth,
duration, boat_id (pseudonymous), photo (optional), zero-catch trips included.
Privacy: locations blurred to grid cell before storage; individual data never
shown to other users.
