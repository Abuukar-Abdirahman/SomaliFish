"""Sea-safety classification for Somali waters. Rules, not machine learning.

Deliberately simple and transparent: a fisherman deciding whether to take a
small boat out deserves a rule he can check himself, not a model output he
cannot interrogate. Thresholds are in src/config.py and are provisional --
they must be reviewed with fishermen (docs/ROADMAP.md Phase 1).

Thresholds use the DAILY MAXIMUM wave height and wind speed, not the mean.
A day averaging 1.2 m that peaks at 2.4 m is not a safe day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config

# Status codes, Somali first (the app's primary language).
SAFE = "AMMAAN"        # safe
CAUTION = "TAXADDAR"   # caution
DANGER = "KHATAR"      # dangerous
UNKNOWN = "LAMA_OGA"   # not known - missing data

LABELS = {
    SAFE:    {"so": "Ammaan",   "en": "Safe",      "color": "#1a9850"},
    CAUTION: {"so": "Taxaddar", "en": "Caution",   "color": "#fee08b"},
    DANGER:  {"so": "Khatar",   "en": "Dangerous", "color": "#d73027"},
    UNKNOWN: {"so": "Lama oga", "en": "Unknown",   "color": "#999999"},
}

# Short advisories shown under the status banner.
ADVICE = {
    SAFE:    {"so": "Xaaladda badda way fiican tahay maanta.",
              "en": "Sea conditions are good today."},
    CAUTION: {"so": "Badda waa kacsan tahay. Doonyaha yaryar ha bixin.",
              "en": "The sea is rough. Small boats should stay in."},
    DANGER:  {"so": "KHATAR: Ha bixin badda maanta.",
              "en": "DANGER: Do not go to sea today."},
    UNKNOWN: {"so": "Xogta badda lama helin. Ka digtoonow.",
              "en": "Sea data unavailable. Be cautious."},
}


def classify(wave_max: pd.Series | np.ndarray,
             wind_max_kmh: pd.Series | np.ndarray) -> pd.Series:
    """Classify sea state from daily maximum wave height (m) and wind (km/h).

    Either condition alone can make a day dangerous -- the worse of the two
    wins. Missing data yields UNKNOWN rather than a guess: telling someone the
    sea is safe on the basis of no data is the one failure mode that could
    get a person killed.
    """
    wave = pd.Series(np.asarray(wave_max, dtype="float64")).reset_index(drop=True)
    wind = pd.Series(np.asarray(wind_max_kmh, dtype="float64")).reset_index(drop=True)

    status = pd.Series(UNKNOWN, index=wave.index, dtype=object)
    known = wave.notna() & wind.notna()

    safe = known & (wave < config.WAVE_SAFE_M) & (wind < config.WIND_SAFE_KMH)
    danger = known & ((wave > config.WAVE_DANGER_M) | (wind > config.WIND_DANGER_KMH))

    status[known] = CAUTION          # the band between safe and dangerous
    status[safe] = SAFE
    status[danger] = DANGER          # applied last: danger overrides
    return status


def summarise(status: pd.Series) -> dict:
    """Area-wide summary for the map's headline banner.

    The headline is the WORST widespread condition, not the average. If a
    quarter of the fishing grounds are dangerous, the banner says dangerous.
    """
    counts = status.value_counts()
    total = int(counts.sum())
    if not total:
        return {"status": UNKNOWN, "counts": {}, "total": 0}

    share = {k: int(v) / total for k, v in counts.items()}
    headline = SAFE
    if share.get(DANGER, 0) >= 0.15:
        headline = DANGER
    elif share.get(DANGER, 0) + share.get(CAUTION, 0) >= 0.30:
        headline = CAUTION
    elif share.get(UNKNOWN, 0) > 0.5:
        headline = UNKNOWN

    return {
        "status": headline,
        "label": LABELS[headline],
        "advice": ADVICE[headline],
        "counts": {k: int(v) for k, v in counts.items()},
        "share_dangerous": round(share.get(DANGER, 0), 3),
        "total": total,
    }
