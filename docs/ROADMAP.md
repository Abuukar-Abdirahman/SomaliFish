# Roadmap

## Phase 0 — Hackathon MVP (now)

Goal: working end-to-end demo.

- [x] Repo, docs, config
- [ ] Download 2019–present ocean data for Somali EEZ (Copernicus + GEBCO)
- [ ] Download GFW fishing effort + OBIS/GBIF species records
- [ ] Build training table (cell × day)
- [ ] Train hotspot XGBoost + validate (temporal holdout, beat baselines)
- [ ] Train 2–3 species models
- [ ] Daily prediction script → GeoJSON heatmap
- [ ] FastAPI + Leaflet web map, Somali labels, safety banner
- [ ] Offline demo fallback data
- [ ] 3-minute pitch: problem → data → live map → honest limits → vision

## Phase 1 — Field validation (1–2 months after)

- Interviews at 1 landing site (Mogadishu), 10–15 fishermen
- Validate species list, safety thresholds, zone naming with fishermen
- Paper/WhatsApp pilot: share daily zone forecast with 5–10 boats, record
  whether it matched reality — first accuracy evidence

## Phase 2 — Data collection app (2–3 months)

- Flutter app: Badda Maanta (conditions + safety) + 30-second catch log
- Offline-first, GPS optional (zone fallback), species by pictures,
  weight by ranges, zero-catch logging
- Pilot: 20–50 boats via cooperative, paid local coordinator
- Target: 500+ verified catch records in 3 months

## Phase 3 — Real model (3–6 months)

- Retrain hotspot model on Somali catch data
- First catch-amount model (LightGBM) if data volume allows
- Accuracy dashboard: predicted vs actual, public and honest

## Phase 4 — Scale

- More landing sites (Kismayo, Bosaso, Berbera...)
- Government/authority dashboard (fisheries pressure, seasonal trends)
- Partnerships: FAO Somalia, World Bank Badmaal project, SIMAD research
- Sustainability features: pressure monitoring, protected-area awareness

## Funding path

Hackathon → university/incubator support → FAO/Badmaal or NGO grant for the
pilot → government contract for national dashboard. The fishermen app stays
free forever; institutions pay for analytics.
