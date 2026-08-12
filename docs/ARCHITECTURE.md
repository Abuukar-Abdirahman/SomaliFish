# Architecture

## Overview

```
┌────────────────────────── DATA LAYER ──────────────────────────┐
│ Copernicus (SST, CHL, currents, waves)   GEBCO (depth)         │
│ Global Fishing Watch (fishing hours)     OBIS/GBIF (species)   │
└───────────────┬────────────────────────────────────────────────┘
                ↓  src/download_data.py (idempotent, per-year)
┌────────────────────────── FEATURE LAYER ───────────────────────┐
│ 0.1° grid over Somali EEZ (~14k ocean cells)                   │
│ One row = cell × day:                                          │
│ sst, sst_gradient, chl, current_speed, wave_height, wind,      │
│ depth, dist_coast, month, lat, lon                             │
└───────────────┬────────────────────────────────────────────────┘
                ↓  src/build_dataset.py
┌────────────────────────── MODEL LAYER ─────────────────────────┐
│ Model A: Hotspot — XGBoost classifier                          │
│   label: fishing presence (GFW hours > threshold)              │
│   output: calibrated score → 0–100 fishing potential           │
│ Model B: Species — XGBoost per species                         │
│   label: OBIS/GBIF presence + sampled pseudo-absences          │
│   output: per-species suitability score                        │
│ Rule module: Safety — thresholds on waves/wind (no ML)         │
└───────────────┬────────────────────────────────────────────────┘
                ↓  src/predict.py (daily batch → GeoJSON)
┌────────────────────────── SERVING LAYER ───────────────────────┐
│ FastAPI: /predictions/today  /zone/{cell_id}  /safety          │
│ Leaflet web map: heatmap + zone popup + safety banner (Somali) │
└────────────────────────────────────────────────────────────────┘
```

## Why XGBoost (v1)

- Tabular features, moderate data size → gradient boosted trees are the
  strongest, cheapest, most debuggable choice.
- Feature importances → explainable to a hackathon jury ("chlorophyll and
  SST fronts drive predictions" — matches fisheries science).
- Deep learning (ConvLSTM etc.) is future work once real catch data exists.

## Feature engineering notes

- **SST gradient (fronts):** compute spatial gradient magnitude of SST per
  cell. Ocean fronts concentrate prey; classic PFZ signal used by India's
  INCOIS advisories.
- **Chlorophyll lag:** fish follow plankton with delay; add chl averaged over
  trailing 7 and 14 days as extra features.
- **Cloud gaps:** L4 gap-filled chlorophyll preferred; else linear temporal
  interpolation up to 3 days, then NaN-flag feature.
- **Month** as cyclic features (sin/cos) to capture monsoon seasonality —
  Somali waters have strong SW (Jun–Sep) / NE (Dec–Mar) monsoon upwelling
  cycles; this matters a lot.

## The monsoon confound (measured, and it shapes everything)

Before any modelling, a check on the 2019 labels found a trap that would have
inverted the model's advice.

Apparent fishing effort by month, against the ocean conditions:

| Month | Fishing rate | Wave (m) | Chlorophyll |
|---|---|---|---|
| Apr–May | 2.0% (peak) | 0.9–1.3 | 0.14–0.22 (lowest) |
| Jul–Aug | 0.01–0.02% | 2.1–2.2 | 0.36–0.46 |
| Sep | 0.35% | 1.8 | 0.51 (highest) |

Fishing effort is **inversely correlated with chlorophyll**. A model trained
naively learns "green water means no fish" — the reverse of fisheries science —
and would steer fishermen away from the sea in its most productive season.

**The cause is fleet absence, not sea state.** In July the entire Somali EEZ
saw 62 vessel-days of apparent fishing against 5,316 in January, a 99%
collapse, while 36,000 navigable cell-days still existed. The distant-water
fleets (Chinese 52%, Iranian 32% of all effort) leave the western Indian Ocean
for the monsoon rather than waiting it out in port.

A per-day "was it navigable?" filter therefore cannot fix this — it was tried
and changed nothing, because the vessels were not present to make a choice.

**Mitigations applied** (`src/baselines.py`):
- `fleet_active_months()` — train only on months when the fleet was present.
  On 2019 this automatically excludes July and August.
- `evaluate_within_month()` — score PR-AUC inside each month and average, so
  a model cannot look good merely by knowing which months are busy.

**Consequence, to be stated openly:** the GFW-trained hotspot model is not
valid in the fleet-dormant SW monsoon months. In that season the map falls
back to the physics index and the safety rules. Those are also the months the
safety module says do not go, so the operational cost is small.

## Validation protocol (non-negotiable)

1. **Temporal holdout:** train ≤ 2023, test 2024+. Never a random split.
2. **Spatial check:** additionally hold out one region (lat band around
   Kismayo, `train_model.SPATIAL_HOLDOUT_LAT`) to test spatial generalization.
   A wider-region variant (train on the western Indian Ocean, test on the
   Somali box) is planned once those downloads exist.
3. **Headline metric is within-month PR-AUC**, not accuracy and not pooled
   PR-AUC. Positives are ~1% of rows, so accuracy is meaningless, and pooled
   PR-AUC rewards knowing the season rather than the place.
4. Compare against two baselines:
   - climatology (each cell-month's historical rate; no ocean data at all)
   - depth-only (bathymetry alone)

   The model must beat both. `predict.py` enforces this: it refuses to use a
   saved model whose metadata does not record `beats_baselines`, and keeps
   ranking zones by the physics index instead.
5. Report metrics honestly in README; jury trust > inflated numbers.

## Why weather is excluded from the hotspot features

Wave height and wind predict fishing effort very well — boats avoid rough seas.
That makes them a weather forecast, not a fish forecast. Since safety is
already a separate transparent rule module, including them would let the model
score well while learning nothing about fish. `train_model.py` excludes them by
default (`--with-weather` to compare).

## Zones: what the map actually outputs

8,294 coloured cells are not advice. `src/zones.py` reduces them to something
a person can act on:

- **Reachability first.** The grid runs ~700 km offshore; an open boat with an
  outboard works within ~50 km of the coast and ~80 km of its landing site.
  Only **606 of 8,294 cells** are reachable. The rest are drawn faint grey —
  visible, never ranked, never recommended. A hotspot beyond range is not
  advice, it is a hazard.
- **Named zones.** Each reachable cell is assigned to its nearest landing site
  (12 sites, Boosaaso → Kismaayo) and a distance band (Xeebta / Dhexe /
  Dibadda). Site × band gives ~36 named zones with a distance and a travel
  time at 18 km/h.
- **Safety outranks score.** Zones are sorted by safety class first, then by
  score. On 2026-08-10 the best water (Xeebta Hobyo, 97/100) ranked *fifth*
  because it was KHATAR, while Qandala at 74/100 ranked first because the Gulf
  of Aden is sheltered from the monsoon swell. Ranking is advice, and advice
  must not send anyone into a storm to reach good fish.

## Coastal wave-data gap

Wave models mask shallow water: **77% of cells within 15 km of the coast had
no wave height** — precisely where artisanal boats work, leaving the safety
layer blind to its own users. Those gaps are now filled from the nearest valid
cell up to 30 km and flagged `wave_estimated`. Filling from offshore
over-warns rather than under-warns, the correct direction for a safety system.
Inshore UNKNOWN fell from 77% to 5%.

## Safety module (rules, not ML)

Simple, transparent thresholds (tune with fishermen input later):

| Condition | Status |
|---|---|
| wave_height < 1.25 m AND wind < 20 km/h | AMMAAN (safe) — green |
| wave_height 1.25–2.0 m OR wind 20–30 km/h | TAXADDAR (caution) — yellow |
| wave_height > 2.0 m OR wind > 30 km/h | KHATAR (dangerous) — red |

Safety overrides everything: a red day shows the warning first, hotspots
dimmed. Never let a high fishing score visually override a danger warning.

## Serving

- Daily cron: download latest ocean fields → predict all cells → write
  `data/predictions/YYYY-MM-DD.geojson` (+ latest.geojson symlink).
- FastAPI serves static GeoJSON + small JSON endpoints; no DB needed for MVP.
- PostGIS enters in v2 when fishermen catch logs arrive.
- Demo fallback: `data/demo/` contains a bundled prediction day so the demo
  works offline.

## v2 (post-hackathon, after fishermen data)

- Replace/augment GFW labels with real Somali catch logs.
- Catch-amount regression (LightGBM) once enough logged trips exist.
- Mobile app (Flutter, offline-first) per docs/ROADMAP.md.
- Retraining pipeline: monthly, with data-quality checks.
