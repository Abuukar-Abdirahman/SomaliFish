# CLAUDE.md — AI assistant instructions for SomaliFish AI

You are helping build **SomaliFish AI**, a potential-fishing-zone prediction system
for Somali waters, for the SIMAD AI & Robotics Hackathon. Read `README.md` and
`docs/` first. This file tells you how to work on this repo.

## Project state

- [x] Repo scaffold, docs, config
- [x] `src/download_data.py` — Copernicus downloader, **tested against the live
      API**. Two dataset families (historical vs NRT), month-chunking + daily
      reduction for sub-daily data, surface-only depth subsetting, `--check`
      mode that validates dataset IDs without an account.
- [x] `src/build_dataset.py` — grid + feature table, tested on NRT data
- [x] `src/download_gfw.py` — GFW fishing effort, 87 months downloaded
- [x] GEBCO depth — block-averaged 24×24 onto the grid; also gives
      `seabed_roughness` and `land_fraction`. Distance-to-coast comes free
      from the SST land mask, no GEBCO needed.
- [x] `src/safety.py` — rule-based sea safety, Somali-first
- [x] `src/zones.py` — reachability, named zones, physics index
- [x] `src/baselines.py` — depth-only + climatology, within-month scoring
- [x] `src/predict.py` — daily JSON output + offline demo fallback
- [x] `src/api.py` + `web/index.html` — FastAPI and Leaflet map, working
- [ ] `src/train_model.py` — **written, blocked on data**: needs processed
      years on both sides of the 2023 split (wind download in progress)
- [ ] Wider western Indian Ocean training region (18× more fishing data)
- [ ] Species models (OBIS/GBIF)

### Facts established by running the pipeline (2026-08-11)

- The Somali EEZ box contains **8,294 ocean cells** at 0.1°, not the ~14,000
  estimated in the docs — the box includes a lot of land. Everything is
  correspondingly faster and smaller than planned.
- Training window is **2019-01-01 → 2026-03-31** (`config.HISTORICAL_END`),
  bounded by reprocessed SST. Train ≤2023 / test 2024+ holds comfortably.
- NRT products publish up to **9 days ahead**, so the map can show a genuine
  short-range forecast, not only the latest observation.
- Chlorophyll runs **7–10 days behind** real time. It is carried forward from
  the last real observation with `chl_age_days` exposed as a feature, and left
  NaN beyond 21 days. Never present carried-forward chlorophyll as current.
- August (SW monsoon) conditions are genuinely dangerous: mean wave height
  2.2 m, mean wind 30 km/h, gusts to 65 km/h. Most of the area is KHATAR by
  the safety thresholds. This is real, not a bug — and it is the season the
  safety feature matters most.
- **The monsoon confound.** Fishing effort is inversely correlated with
  chlorophyll, because the foreign fleet leaves the region entirely during the
  SW monsoon (July: 62 vessel-days vs January's 5,316). Training naively would
  teach the model that the richest water is the worst. Fixed by excluding
  fleet-dormant months and scoring within-month; see docs/ARCHITECTURE.md.
  A per-day navigability filter does NOT fix this — it was tried and failed.
- **Only 606 of 8,294 cells are reachable** by an artisanal boat (≤50 km
  offshore, ≤80 km from a landing site). Never rank or recommend the rest.
- GFW flag composition, 2019–2026: CHN 52.5%, IRN 32.0%, TWN 4.2%,
  **SOM 1.4%**. Use these exact numbers in the pitch — they are the honest
  evidence for the AIS limitation, and they are striking.
- Joins on lat/lon floats silently fail (float32 vs float64 mismatch dropped
  96% of labels once). Always join on `lat_idx`/`lon_idx` integers.

## Honesty machinery already built — do not weaken it

- `predict.py` refuses to use a saved model whose metadata lacks
  `beats_baselines`, and falls back to the physics index.
- The physics index is labelled "measured indicator, not a prediction" in
  Somali and English wherever it appears.
- `chl_age_days` and `wave_estimated` expose where inputs are stale or filled.
- Zone ranking sorts by safety class before score.
- Missing wave/wind yields LAMA_OGA (unknown), never AMMAAN (safe).

Work through unchecked items in order unless the user says otherwise.

## Hard constraints — do not violate

1. **Never fabricate data.** If a download fails or a dataset is unavailable,
   say so and stop. Do not generate synthetic ocean or catch data and pass it
   off as real. (Synthetic data for unit tests is fine if clearly labeled.)
2. **No fake precision.** v1 must NOT output "expected catch: 47 kg" or made-up
   percentages presented as validated probabilities. Outputs are a relative
   fishing-potential score (0–100) and clearly labeled model confidence.
3. **Temporal validation only.** Train on earlier years, test on later years
   (e.g. train ≤2023, test 2024+). Never random train/test split — spatial and
   temporal autocorrelation makes random splits meaninglessly optimistic.
4. **Small-boat honesty.** Global Fishing Watch mostly tracks industrial vessels
   (AIS). Somali artisanal boats are largely invisible in it. Keep this
   limitation documented in outputs and docs.
5. **Keep the stack minimal.** Python, pandas, xarray, copernicusmarine,
   scikit-learn, xgboost, fastapi, leaflet. No PyTorch/deep learning in v1.
   Don't add dependencies without clear need.

## Technical decisions already made

- **Study area:** Somali EEZ bounding box, lat -2.0 → 12.5, lon 40.5 → 52.5
  (defined in `src/config.py`). Grid resolution 0.1° (~11 km).
- **Features per cell-day:** SST (°C), SST gradient (fronts), chlorophyll-a
  (mg/m³), current speed (m/s), wave height (m), wind speed (m/s), depth (m),
  distance to coast (km), month, lat, lon.
- **Labels v1:** fishing presence/intensity from Global Fishing Watch API
  (apparent fishing hours per cell). Binary presence first; intensity later.
- **Model v1:** XGBoost binary classifier (fishing / no fishing given
  conditions), then calibrated score → 0–100 "fishing potential".
- **Species model:** separate XGBoost per major species (tuna, kingfish,
  snapper, sardine) using OBIS/GBIF presence records + pseudo-absences.
  Species list may be refined after fishermen interviews.
- **Serving:** precompute daily predictions for the whole grid (it's small —
  ~14,000 cells), store as GeoJSON, serve via FastAPI, render in Leaflet.

## Data access summary

Full details in `docs/DATA_SOURCES.md`. Short version:

| Data | Source | Access |
|---|---|---|
| SST, chlorophyll, currents, waves | Copernicus Marine | `copernicusmarine` Python package, free account |
| Depth | GEBCO 2024 grid | one-time NetCDF download |
| Fishing activity | Global Fishing Watch | REST API, free token |
| Species records | OBIS / GBIF | `pyobis` / REST API, no auth |

## Coding conventions

- Python 3.10+, type hints, docstrings on public functions.
- All paths/constants in `src/config.py`, never hardcoded.
- Scripts must be runnable as `python src/<name>.py` with sensible defaults
  and `--help` (use argparse).
- Downloads must be resumable/idempotent: skip files that already exist.
- Handle cloud-gap NaNs in chlorophyll explicitly (interpolate or flag).
- Keep memory in mind: process satellite data year by year, not all at once.

## When the user asks for the web map

- Leaflet + plain JS or minimal React, heatmap layer from GeoJSON scores.
- Somali-first labels: "Badda Maanta" (today's sea), "Rajo fiican" (promising),
  "Khatar" (dangerous). English secondary.
- Include a safety banner driven by wave height + wind thresholds
  (see `docs/ARCHITECTURE.md` §Safety).
- Must degrade gracefully on cheap Android phones (small assets, no heavy libs).

## Hackathon context

The demo that must work end-to-end: open web map → see today's (or a cached
recent day's) fishing-potential heatmap for Somali waters → click a zone →
see score, likely species, safety status in Somali. If live APIs fail during
demo, fall back to bundled cached data in `data/demo/` — build that fallback.
