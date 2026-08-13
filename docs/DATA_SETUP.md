# Data setup — what to download, and where it goes

Everything here is free. Nothing is committed to git except the demo day.

**You almost certainly do not need all of it.** Pick a tier:

| Tier | Size | You get | Accounts needed |
|---|---|---|---|
| **0. Demo** | 0 MB | Working map, real data, bundled | none |
| **1. Live map** | ~40 MB | Today's conditions, refreshable | Copernicus |
| **2. Training** | ~2.9 GB | Retrain and validate the model | Copernicus + GFW + GEBCO |
| **3. Wide region** | ~5.5 GB | Reproduce the transfer experiment | same as tier 2 |

Total if you download everything: **8.3 GB raw + 1.6 GB processed**.

---

## Tier 0 — just run the demo (no downloads, no accounts)

`data/demo/latest.json` is committed. It is a real prediction day
(2026-08-10) with 8,294 cells and 36 ranked zones.

```bash
pip install -r requirements.txt
python src/api.py         # http://127.0.0.1:8000
```

This is also the hackathon fallback — if the venue network dies, the map still
works and labels itself as cached.

---

## Accounts (only for tier 1+)

### Copernicus Marine — ocean data

1. Register at <https://data.marine.copernicus.eu> and **confirm the email**.
2. `copernicusmarine login` — it asks for your **username, not your email**.
   Stored once in `~/.copernicusmarine/`, outside the repo.

Check the dataset IDs resolve before downloading anything. This needs **no
account** and is worth running first, because Copernicus renames products
between versions:

```bash
python src/download_data.py --check
python src/download_data.py --check --mode nrt
```

### Global Fishing Watch — training labels (tier 2+ only)

Free token from <https://globalfishingwatch.org/our-apis/>. Choose
**Research / Academic, non-commercial**. Then:

```bash
cp .env.example .env      # paste the token into .env; it is gitignored
python src/download_gfw.py --check
```

---

## Tier 1 — live map (~40 MB, ~2 minutes)

```bash
python src/download_data.py --mode nrt --days 5
python src/build_dataset.py --mode nrt
python src/predict.py --demo
python src/api.py
```

Lands in `data/raw/nrt/`:

| Folder | Size | Product |
|---|---|---|
| `sst/` | 0.4 MB | `METOFFICE-GLO-SST-L4-NRT-OBS-SST-V2` |
| `chl/` | 16 MB | `cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D` |
| `cur/` | 0.8 MB | `cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m` |
| `thetao/` | 20 MB | `cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m` |
| `wav/` | 1.6 MB | `cmems_mod_glo_wav_anfc_0.083deg_PT3H-i` |
| `wnd/` | 1.1 MB | `cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H` |

Chlorophyll publishes ~7–10 days behind real time, so the downloader
automatically pulls a longer tail and carries the last real observation
forward, recording `chl_age_days`.

---

## Tier 2 — training data (~2.9 GB, several hours)

### 2a. Ocean fields

```bash
python src/download_data.py --mode historical
```

Lands in `data/raw/historical/`:

| Folder | Files | Size | Product | Coverage |
|---|---|---|---|---|
| `sst/` | 8 | 352 MB | `METOFFICE-GLO-SST-L4-REP-OBS-SST` | 1981-10 → 2026-03 |
| `chl/` | 8 | 1013 MB | `cmems_obs-oc_glo_bgc-plankton_my_l4-gapfree-multi-4km_P1D` | 1997-09 → 2026-08 |
| `phy/` | 8 | 385 MB | `cmems_mod_glo_phy_my_0.083deg_P1D-m` | 1993-01 → 2026-06 |
| `wav/` | 87 | 180 MB | `cmems_mod_glo_wav_my_0.2deg_PT3H-i` | 1980-01 → 2026-05 |
| `wnd/` | 87 | 964 MB | `cmems_obs-wind_glo_phy_my_l4_0.125deg_PT1H` | 2007-01 → 2026-04 |

One file per year, except waves and wind which are **one file per month** —
they are sub-daily and get reduced to daily mean/max immediately after each
chunk downloads. Raw hourly wind would otherwise be ~800 MB per year.

**Wind is the slow one** (87 monthly requests). Expect a few hours. Downloads
are idempotent, so a killed run resumes — just run it again.

### 2b. Fishing-effort labels

```bash
python src/download_gfw.py            # 87 files → data/raw/gfw/, 2.2 MB
python src/download_gfw.py --summary  # who is actually fishing out there
```

Tiny, fast, and the `--summary` output is worth reading: China 52.5%,
Iran 32.0%, Taiwan 4.2%, **Somalia 1.4%**. That is the AIS limitation in one
table.

### 2c. GEBCO bathymetry — **MANUAL, no API**

This is the only step that cannot be scripted.

1. Go to <https://download.gebco.net>
2. Enter the bounding box:
   - **North 12.5, South −2.0, West 40.5, East 52.5**
3. Format: **2D netCDF** (not the global grid — you want the cutout, ~20 MB,
   not 7 GB)
4. Download and put the file in **`data/raw/gebco/`**

```
data/raw/gebco/gebco_2026_n12.5_s-2.0_w40.5_e52.5.nc     # 20 MB
```

Any `.nc` filename works — it is found by glob. It is block-averaged 24×24
onto the 0.1° grid, giving `depth`, `seabed_roughness` and `land_fraction`.

You can build and run without it; depth is simply skipped, with a warning.
But depth is one of the two required validation baselines, so you need it to
reproduce the model results.

### 2d. Build and train

```bash
python src/build_dataset.py                   # 8 files → data/processed/, ~1.1 GB
python src/baselines.py
python src/train_model.py --spatial-holdout
```

Each year is ~3.03M rows × 28 columns, ~145 MB of Parquet.

---

## Tier 3 — wide region, the transfer experiment (~5.5 GB)

Only if you want to reproduce the negative transfer result in the README.
**We already ran this and it failed** — training on 7.8× more data made the
model worse. Reproduce it to check us, not because it is a promising path.

```bash
python src/download_data.py --mode historical --region wide --training-only \
    --start 2019-01-01 --end 2026-03-31
python src/download_gfw.py --region wide --start 2019-01-01 --end 2026-03-31
```

Lands in `data/raw/region_wide/` (kept entirely separate so it can never
contaminate the Somali data):

| Folder | Files | Size |
|---|---|---|
| `historical/sst/` | 8 | 1.1 GB |
| `historical/chl/` | 8 | 3.2 GB |
| `historical/phy/` | 8 | 1.2 GB |
| `gfw/` | 87 | 13 MB |

Note `--training-only`: waves and wind are **excluded from the model by
design** (they predict weather, not fish), so the wide region does not need
them — which also skips the two slowest datasets.

There is **no GEBCO for the wide box**, so the transfer experiment runs
without depth, and the Somali control is refitted on the same depth-free
features to keep the comparison fair.

```bash
for y in 2019 2020 2021 2022 2023; do
  python src/build_dataset.py --region wide --year $y --neg-per-pos 20
done
python src/train_model.py --transfer
```

`--neg-per-pos 20` is **required** here: the wide grid is 55,000 cells, so a
full year is ~20M rows and will not fit in memory. It keeps every positive and
thins negatives. Never use it on data you intend to evaluate on — it changes
the base rate.

---

## Where everything ends up

```
data/
├── demo/          ← COMMITTED. One real prediction day, the offline fallback.
├── raw/           ← gitignored, 8.3 GB if you take everything
│   ├── historical/{sst,chl,phy,wav,wnd}/     Somali box, training
│   ├── nrt/{sst,chl,cur,thetao,wav,wnd}/     Somali box, today
│   ├── gfw/                                  fishing effort labels
│   ├── gebco/                                bathymetry (MANUAL download)
│   └── region_wide/                          wide region, training only
│       ├── historical/{sst,chl,phy}/
│       └── gfw/
├── processed/     ← gitignored, ~1.6 GB. One Parquet per year.
│   ├── features_historical_YYYY.parquet          8 files, Somali
│   ├── features_wide_historical_YYYY.parquet     5 files, wide
│   └── features_nrt_latest.parquet
└── predictions/   ← gitignored. Daily JSON output.
```

Check what you actually have at any time:

```bash
python src/build_dataset.py --list
```

---

## Study area

All Somali-box downloads use, from `src/config.py`:

- **Latitude −2.0 → 12.5, longitude 40.5 → 52.5**, at **0.1°** (~11 km)
- 145 × 120 = 17,400 grid cells, of which **8,294 are ocean**
- Of those, **606 are reachable** by an artisanal day-boat

The wide region is lat −10 → 15, lon 38 → 60 (250 × 220 = 55,000 cells).

Change the box in `config.REGIONS` if you want a different area — the grid,
downloads, zoning and distance-to-coast all derive from it.

---

## Troubleshooting

**"Invalid credentials" from Copernicus** — you almost certainly typed your
email. It wants the username you chose at registration. Also confirm the
email before first login.

**A dataset ID stops resolving** — Copernicus renamed the product.
`python src/download_data.py --check` shows which one and what the catalogue
has now; update `config.DATASETS_HISTORICAL` or `DATASETS_NRT`.

**Download died part-way** — just re-run it. Existing files are skipped.

**MemoryError during download** — a whole year of a model product can exceed
available memory; the downloader catches this and retries the span month by
month automatically.

**MemoryError during build** — use `--neg-per-pos 20`, or build one year at a
time with `--year`. The build streams in 15-day chunks, but the final table
still has to fit.

**`[MISSING SOURCES]` when building** — a download failed silently for that
year. The message names the dataset and prints the exact command to fix it.
Do not ignore it: a year built without a source is missing whole feature
columns, and nothing downstream will complain.
