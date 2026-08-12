# Contributing to SomaliFish AI

Where the project stands, what you need to run it, and what is worth working
on. Read `README.md` first for the honest results, and
`docs/ARCHITECTURE.md` for the design.

---

## Run the demo in 2 minutes — no accounts needed

The repo ships a real prediction day in `data/demo/`, so you can see the whole
thing working before signing up for anything.

```bash
git clone https://github.com/Abuukar-Abdirahman/SomaliFish.git
cd SomaliFish

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python src/api.py
```

Open <http://127.0.0.1:8000>. You should see the Somali coast, a red **KHATAR**
banner, 36 ranked fishing zones, and four map layers. That data is real
Copernicus satellite data for 2026-08-10, bundled as the offline fallback.

**This is also the hackathon demo path.** If the venue wifi dies, the map still
works — it just labels the data as cached.

---

## Run the full pipeline — needs two free accounts

Only required if you want to regenerate data or retrain.

### 1. Copernicus Marine (ocean data)

Register at <https://data.marine.copernicus.eu>, confirm the email, then:

```bash
copernicusmarine login       # asks for USERNAME (not email), stores a config file
```

Verify every dataset ID still resolves — this needs no account and is worth
running first, because Copernicus renames products between versions:

```bash
python src/download_data.py --check
python src/download_data.py --check --mode nrt
```

### 2. Global Fishing Watch (training labels)

Free token from <https://globalfishingwatch.org/our-apis/> — choose
**Research / Academic, non-commercial**. Then:

```bash
cp .env.example .env
# paste your token into .env
python src/download_gfw.py --check
```

### 3. GEBCO bathymetry (manual, one file)

There is no API. Go to <https://download.gebco.net>, request the box
**lat −2 → 12.5, lon 40.5 → 52.5**, format **2D netCDF**, and drop the file
into `data/raw/gebco/`. It is picked up automatically.

### 4. Download, build, predict

```bash
# Today's data for the map (fast, ~1 minute)
python src/download_data.py --mode nrt --days 5
python src/build_dataset.py --mode nrt
python src/predict.py --demo
python src/api.py

# Full training archive (SLOW: ~2.9 GB, several hours)
python src/download_data.py --mode historical
python src/download_gfw.py
python src/build_dataset.py                  # one Parquet per year
python src/baselines.py                      # depth-only + climatology
python src/train_model.py --spatial-holdout
```

**Downloads are idempotent** — existing files are skipped, so a killed download
resumes safely. If one dies part-way, just run it again.

---

## Pipeline at a glance

```
download_data.py   Copernicus ocean fields  (historical = training, nrt = today)
download_gfw.py    fishing effort = labels
        ↓
build_dataset.py   0.1° grid, features, GEBCO depth, GFW labels  → Parquet/year
        ↓
train_model.py     XGBoost + temporal & spatial holdout   ┐
baselines.py       depth-only, climatology, physics index ┘ → metrics
        ↓
predict.py         daily JSON + offline demo fallback
        ↓
api.py + web/      FastAPI and Leaflet map
```

| File | Does |
|---|---|
| `src/config.py` | every constant: bbox, regions, thresholds, dataset IDs |
| `src/download_data.py` | Copernicus downloader, `--check`, `--region` |
| `src/download_gfw.py` | GFW effort, `--summary` shows who is fishing |
| `src/build_dataset.py` | grid, features, depth, labels |
| `src/baselines.py` | the bar the model must clear; confound filters |
| `src/train_model.py` | training + honest validation |
| `src/zones.py` | reachability, named zones, physics index |
| `src/safety.py` | rule-based sea safety (no ML) |
| `src/predict.py` | daily output |
| `src/api.py`, `web/index.html` | serving and map |

---

## Current status

**Working and tested**

- Copernicus downloader, both dataset families, verified against the live API
- GFW labels: 87 months, 163,672 vessel-day rows for Somali waters
- Feature pipeline: 8 years built, ~3.03M rows per year, 28 columns
- GEBCO depth, block-averaged 24×24, agrees 99.63% with the SST land mask
- Safety module, zones, reachability, physics index
- FastAPI + Leaflet map with interpolated rendering
- Offline demo fallback

**Known-failing, on purpose**

The hotspot model **does not beat a climatology baseline**, and has **zero
spatial transfer** (lift 1.00 on a held-out region — pure chance). See the
README results table. `predict.py` refuses to serve an unvalidated model and
falls back to the physics index. Do not "fix" this by removing the gate.

**In progress**

- Wider western Indian Ocean training region (`--region wide`): data
  downloaded (3.3 GB ocean, 1.1M fished cell-days), needs `build_dataset.py`
  to grow a `--region` flag before it can be trained.

**Not started**

- Species models (OBIS/GBIF) — `config.TARGET_SPECIES` is defined, nothing else
- Somali-language review of every string by an actual Somali speaker
- Mobile app (roadmap Phase 2)

---

## Good first issues

1. **`build_dataset.py --region wide`** — needs the grid and paths to honour
   `config.set_region()`, like the downloaders already do. Unblocks the
   transfer experiment.
2. **Somali language review.** Strings live in `src/safety.py` (`LABELS`,
   `ADVICE`), `src/zones.py` (`rating`, band names) and `web/index.html`.
   Written by a non-speaker and certainly clumsy in places.
3. **Safety thresholds** in `config.py` are guesses from general small-boat
   guidance. They have never been checked with a Somali fisherman. This is the
   single highest-value validation in the project.
4. **Landing site coordinates** in `config.LANDING_SITES` come from public
   geography, not field knowledge. Real launch points may differ by kilometres.
5. **Species presence models** using OBIS/GBIF, following
   `docs/DATA_SOURCES.md` §4.

---

## House rules

These come from `CLAUDE.md` and exist because a wrong answer here can put
someone at sea in a storm.

1. **Never fabricate data.** If a download fails, say so and stop. Synthetic
   data is fine in tests, clearly labelled.
2. **No fake precision.** No "expected catch 47 kg". Outputs are a relative
   0–100 index plus clearly labelled confidence.
3. **Temporal validation only.** Train on earlier years, test on later ones.
   A random split is meaningless here — ocean fields are autocorrelated in
   space and time, so it puts near-duplicate rows on both sides.
4. **Report the AIS limitation everywhere.** GFW sees industrial and foreign
   vessels. Somali-flagged boats are 1.4% of effort in Somali waters.
5. **Safety outranks everything.** Missing wave or wind data gives
   `LAMA_OGA` (unknown), never `AMMAAN` (safe). Zone ranking sorts by safety
   class before score. Never let a high fishing score visually outshout a
   danger warning.
6. **Join on `lat_idx`/`lon_idx`, never on lat/lon floats.** `float32(-1.85)`
   is not equal to `float64(-1.85)`; a float join silently dropped 96% of the
   labels once and nothing errored.
7. **Keep the stack small.** Python, pandas, xarray, copernicusmarine,
   scikit-learn, xgboost, fastapi, leaflet. No deep learning in v1. The map
   must stay usable on a cheap Android phone.

---

## Testing your changes

There is no test suite yet (a real gap — contributions welcome). Minimum
manual check before opening a PR:

```bash
python src/download_data.py --check        # dataset IDs still resolve
python src/build_dataset.py --list         # what data you actually have
python src/build_dataset.py --mode nrt     # rebuild features
python src/predict.py --demo               # regenerate output
python src/api.py                          # map still loads, zones still rank
```

If you touched the model or validation, also run:

```bash
python src/train_model.py --dry-run        # data readiness
python src/baselines.py                    # baselines still compute
```

Sanity-check the numbers rather than trusting them. Several real bugs in this
repo were caught only because two figures that should have agreed did not.
