# SomaliFish AI

**AI-Based Potential Fishing Zone Prediction for Somali Waters Using Satellite Ocean Data**

Built for the SIMAD AI Institute — AI & Robotics Hackathon (Mogadishu, 2026).

## The problem

Somalia has one of the longest coastlines in Africa (~3,300 km) and rich fishing grounds, but small-scale Somali fishermen have no scientific tools to decide **where** and **when** to fish, or whether the sea is **safe** today. They rely on experience and luck, burning fuel searching for fish, and sometimes going out in dangerous conditions.

Meanwhile, satellites measure the Somali ocean every day for free — sea temperature, chlorophyll (plankton, the base of the fish food chain), currents, wind, and waves. Countries like India already use this data to publish "Potential Fishing Zone" advisories for their fishermen. Nobody has built this for Somalia.

## What SomaliFish AI does

An AI system that combines free satellite ocean data with fishing-activity data to predict, for every ~11 km cell of Somali waters, each day:

1. **Fishing potential score** — how favorable conditions are for productive fishing (heatmap)
2. **Likely species** — e.g. tuna, kingfish, snapper (from marine species observation databases)
3. **Safety status** — waves/wind based go/no-go recommendation, in Somali

Output is shown on an interactive web map (MVP) and later a Somali-language, offline-first mobile app for fishermen.

## Honest scope (v1)

- We have **no historical Somali catch logs** (they don't exist yet — that is part of the problem we're solving).
- v1 is therefore trained on **fishing activity** from Global Fishing Watch (where vessels chose to fish) + ocean conditions, and species presence from **OBIS/GBIF** scientific records.
- v1 predicts *potential fishing zones*, not catch weight in kg. Catch prediction is future work, enabled by our planned fishermen data-collection app (see `docs/ROADMAP.md`).

## Architecture (v1)

```
Copernicus Marine (SST, chlorophyll, currents, waves)
        +
GEBCO bathymetry (depth)
        +
Global Fishing Watch (fishing activity = labels)
        +
OBIS / GBIF (species observations)
        ↓
Feature table: one row = one grid cell × one day
        ↓
XGBoost hotspot model  +  species presence model
        ↓
Daily prediction for every cell in the Somali EEZ grid
        ↓
FastAPI backend → Leaflet web map (heatmap + safety)
```

See `docs/ARCHITECTURE.md` for details.

## Repo structure

```
somalifish-ai/
├── README.md            ← you are here
├── CLAUDE.md            ← instructions for AI coding assistants
├── requirements.txt
├── docs/
│   ├── ARCHITECTURE.md  ← system design, models, validation strategy
│   ├── DATA_SOURCES.md  ← every dataset, exact access method, variables
│   └── ROADMAP.md       ← phases from MVP to national platform
├── src/
│   ├── config.py        ← Somali EEZ bounding box, grid resolution, constants
│   ├── download_data.py ← pulls satellite data (Copernicus) — STEP 1
│   ├── build_dataset.py ← merges sources into training table — STEP 2 (TODO)
│   ├── train_model.py   ← XGBoost training + validation — STEP 3 (TODO)
│   └── predict.py       ← daily heatmap generation — STEP 4 (TODO)
├── notebooks/           ← exploration notebooks
├── data/                ← raw + processed data (gitignored)
└── models/              ← trained model files (gitignored)
```

## Quick start

```bash
python -m venv .venv && .venv/Scripts/activate     # Windows
pip install -r requirements.txt

# 1. Free account at https://data.marine.copernicus.eu, then:
copernicusmarine login

# 2. Free GFW API token from https://globalfishingwatch.org/our-apis/
#    Save it to .env as:  GFW_API_TOKEN=<token>

# 3. Verify every dataset ID resolves (no account needed):
python src/download_data.py --check

# 4. Download. Historical = training, NRT = today's map.
python src/download_data.py --mode historical      # ~2.5 GB, slow
python src/download_data.py --mode nrt --days 5
python src/download_gfw.py                         # fishing-effort labels

# 5. GEBCO depth: manual download of the bounding-box cutout from
#    https://download.gebco.net (2D netCDF) into data/raw/gebco/

# 6. Build features, predict, serve:
python src/build_dataset.py --mode nrt
python src/predict.py --demo
python src/api.py            # http://127.0.0.1:8000
```

## Results so far — and an honest negative one

**The trained hotspot model does not beat a climatology baseline, and it does
not generalise to water it has not seen.**

Full temporal holdout: train 2019–2023, test 2024 – 2026-Q1. 5.78M unsampled
test rows, base rate 0.34%. Headline metric is within-month PR-AUC (accuracy
is meaningless at this class balance, and pooled PR-AUC rewards merely knowing
which months are busy).

| Model | Within-month PR-AUC | Lift over base rate |
|---|---|---|
| **climatology** (cell × month history, no ocean data) | **0.0145** | **4.89×** |
| hotspot (XGBoost on ocean conditions) | 0.0096 | 3.36× |
| depth-only | 0.0072 | 1.92× |
| physics index (what the map ships) | 0.0060 | 2.03× |

**Spatial holdout — the decisive result.** Retraining with the Kismayo lat
band (−2.0 → 1.5) withheld entirely, then testing on it:

| | Within-month PR-AUC | Lift |
|---|---|---|
| same region (seen in training) | 0.0096 | 3.36× |
| **unseen region** | **0.0042** | **1.00×** |

A lift of 1.00 is exactly chance. **The model has no spatial skill whatsoever
in water it was not trained on.** It works only where it has already seen
fishing, which means it is recalling a map, not learning how the ocean
produces fish. Feature importances agree: month 28.7%, longitude 15.3%,
depth 11.7% — season and place, with chlorophyll not appearing until 6.6%.

Consequently **the map does not show a model prediction.** `src/predict.py`
refuses to load a model whose metadata lacks `beats_baselines` and falls back
to the physics index. That gate is enforced in code, not by convention.

**The physics index is not validated either.** Scored on the same held-out
data it ranks last of the four. Two caveats: its average *lift* (2.03×) beats
depth-only's (1.92×), and more importantly it is not trying to predict vessel
behaviour — it claims to indicate biologically productive water, and no
dataset we have measures that. So this is not evidence the index is wrong; it
is an absence of evidence that it is right. We ship it because it is
transparent and physically motivated, and we label it a *measured indicator,
not a prediction*, everywhere it appears.

Likeliest explanation for all of the above: **AIS fishing effort is a weak
proxy for fish.** Industrial vessels choose grounds by licences, quotas, fuel
cost and accumulated crew knowledge, not by yesterday's chlorophyll. Training
on the wider western Indian Ocean (7.8× the fishing data — 1,145,721 fished
cell-days across 35,586 cells) is the next experiment, though the spatial
holdout result sets a low prior on it succeeding.

## What is real, today

- Daily ocean fields for the Somali EEZ from Copernicus, 2019 → present
- 8,294 ocean cells at 0.1°; **606** reachable by an artisanal day-boat
- 36 named, ranked fishing zones with distance and travel time
- Rule-based sea-safety in Somali (AMMAAN / TAXADDAR / KHATAR)
- Working FastAPI + Leaflet map, offline demo fallback

## Known limitations (please read before believing anything)

1. **GFW labels are industrial and foreign vessels.** Over 2019–2026 in Somali
   waters: China 52.5%, Iran 32.0%, Taiwan 4.2%, **Somalia 1.4%**. Somali
   artisanal boats carry no AIS and are essentially invisible.
2. **The monsoon confound.** Fishing effort is *inversely* correlated with
   chlorophyll, because the foreign fleet leaves the region entirely in the SW
   monsoon (July: 62 vessel-days vs January's 5,316). Fleet-dormant months are
   excluded from training; the model would otherwise learn that the richest
   water is the worst.
3. **No model prediction in the off-season**, by design.
4. **Chlorophyll runs 7–10 days behind** real time; it is carried forward with
   `chl_age_days` exposed, and left blank beyond 21 days.
5. **77% of inshore cells have no measured wave height** — wave models mask
   shallow water. Filled from the nearest cell within 30 km and flagged
   `wave_estimated`; the fill over-warns rather than under-warns.
6. **Safety thresholds are provisional** and have never been reviewed by a
   Somali fisherman. That is Phase 1 of the roadmap.
7. **No catch data exists.** Nothing here predicts kilograms.

## Team

[Add team member names, university, contact]

## License

MIT
