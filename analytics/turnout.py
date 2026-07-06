"""
turnout.py
----------
Predicts cumulative voter turnout (%) at a polling booth given time-of-day
and a few contextual features.

The EMS front-end only ever has same-day report data in localStorage, so for
a meaningful regression we bootstrap a synthetic-but-realistic historical
training set (typical Indian-election-style turnout curves across booth
types/weather/day-of-week) and train a small RandomForestRegressor on it.
If real historical CSVs are supplied via /api/analytics/turnout/train, those
are used instead — see `train_from_dataframe`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

BOOTH_TYPES = ["urban", "semi_urban", "rural"]
WEATHER = ["clear", "rain", "extreme_heat"]

_FEATURE_COLUMNS = ["hour", "booth_type_code", "weather_code", "is_weekend", "prior_election_turnout"]

_model: RandomForestRegressor | None = None
_metrics: dict = {}


def _encode_booth_type(x: str) -> int:
    return BOOTH_TYPES.index(x) if x in BOOTH_TYPES else 0


def _encode_weather(x: str) -> int:
    return WEATHER.index(x) if x in WEATHER else 0


def generate_synthetic_training_data(n_booths: int = 400, seed: int = 42) -> pd.DataFrame:
    """
    Builds a synthetic historical dataset: for each simulated booth on a past
    election day, cumulative turnout % sampled every 2 hours from 7AM-7PM.
    Turnout curves are shaped to look like typical real-world patterns:
    fast morning rush, midday lull, evening rush, with variation by booth
    type, weather, and weekend/weekday.
    """
    rng = np.random.default_rng(seed)
    hours = list(range(7, 20, 2))  # 7,9,11,13,15,17,19
    rows = []

    for booth_id in range(n_booths):
        booth_type = rng.choice(BOOTH_TYPES, p=[0.4, 0.35, 0.25])
        weather = rng.choice(WEATHER, p=[0.7, 0.2, 0.1])
        is_weekend = int(rng.random() < 0.3)
        prior_turnout = float(np.clip(rng.normal(65, 12), 30, 95))

        # base final turnout influenced by booth type/weather/weekend
        base_final = prior_turnout + rng.normal(0, 5)
        base_final += {"urban": -3, "semi_urban": 0, "rural": 4}[booth_type]
        base_final += {"clear": 2, "rain": -6, "extreme_heat": -4}[weather]
        base_final += 3 if is_weekend else -2
        base_final = float(np.clip(base_final, 20, 95))

        # shape a monotonic S-curve across the day reaching base_final by 7PM
        curve_shape = rng.normal(1.0, 0.1)
        for h in hours:
            t = (h - 7) / 12.0  # 0..1 across the day
            progress = 1 / (1 + np.exp(-6 * curve_shape * (t - 0.35)))  # sigmoid rush
            cumulative = base_final * progress
            cumulative += rng.normal(0, 1.5)
            cumulative = float(np.clip(cumulative, 0, base_final))

            rows.append({
                "hour": h,
                "booth_type": booth_type,
                "weather": weather,
                "is_weekend": is_weekend,
                "prior_election_turnout": round(prior_turnout, 1),
                "cumulative_turnout_pct": round(cumulative, 2),
            })

    return pd.DataFrame(rows)


def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["booth_type_code"] = out["booth_type"].apply(_encode_booth_type)
    out["weather_code"] = out["weather"].apply(_encode_weather)
    return out


def train_from_dataframe(df: pd.DataFrame) -> dict:
    """
    df must contain columns: hour, booth_type, weather, is_weekend,
    prior_election_turnout, cumulative_turnout_pct
    """
    global _model, _metrics

    prepared = _prepare_features(df)
    X = prepared[_FEATURE_COLUMNS]
    y = prepared["cumulative_turnout_pct"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = float(mean_absolute_error(y_test, preds))

    _model = model
    _metrics = {
        "mae": round(mae, 2),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importances": dict(zip(_FEATURE_COLUMNS, [round(v, 3) for v in model.feature_importances_])),
    }
    return _metrics


def ensure_trained() -> dict:
    """Train on synthetic data if no model has been trained yet."""
    global _model
    if _model is None:
        df = generate_synthetic_training_data()
        return train_from_dataframe(df)
    return _metrics


def predict(records: list) -> list:
    """
    records: list of dicts with hour, booth_type, weather, is_weekend,
    prior_election_turnout
    Returns list of {..., predicted_turnout_pct}
    """
    ensure_trained()
    df = pd.DataFrame(records)

    defaults = {"weather": "clear", "is_weekend": 0, "prior_election_turnout": 65.0}
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        else:
            df[col] = df[col].fillna(val)

    if "hour" not in df.columns or "booth_type" not in df.columns:
        raise ValueError("Each record needs at least 'hour' and 'booth_type'.")

    prepared = _prepare_features(df)
    X = prepared[_FEATURE_COLUMNS]
    preds = _model.predict(X)

    df["predicted_turnout_pct"] = np.round(preds, 2)
    return df.to_dict(orient="records")


def model_metrics() -> dict:
    ensure_trained()
    return _metrics
