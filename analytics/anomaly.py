"""
anomaly.py
----------
Flags suspicious voting-area patterns using an IsolationForest over simple,
explainable per-area features built from citizen issues + observer reports:

  - report_count        total reports for the area
  - negative_ratio       share of reports with negative sentiment
  - category_diversity   how many distinct issue categories were seen
  - photo_ratio          share of reports that included photo evidence
  - report_velocity      reports per hour span covered by the area's reports

Small feature set on purpose: this is meant to triage/prioritize which
booths an admin should look at first, not to make an automated accusation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .sentiment import score_dataframe

CATEGORY_KEYWORDS = {
    "Security": ["security", "crowd", "violence", "threat", "intimidat"],
    "Machines": ["machine", "evm", "voting machine", "error", "malfunction"],
    "Booth Clash": ["clash", "fight", "conflict"],
    "Bribing": ["bribe", "bribing", "money", "payment", "corrupt"],
    "Technical": ["technical", "network", "internet", "power", "electric"],
}


def infer_category(text: str) -> str:
    if not text:
        return "Other"
    t = text.lower()
    for cat, keys in CATEGORY_KEYWORDS.items():
        if any(k in t for k in keys):
            return cat
    return "Other"


def build_area_features(unified_df: pd.DataFrame) -> pd.DataFrame:
    if unified_df.empty:
        return pd.DataFrame(
            columns=["area", "report_count", "negative_ratio", "category_diversity",
                     "photo_ratio", "report_velocity_per_hr"]
        )

    df = unified_df.copy()
    df["category"] = df["text"].apply(infer_category)
    df = score_dataframe(df, text_col="text")

    rows = []
    for area, g in df.groupby("area"):
        report_count = len(g)
        negative_ratio = (g["sentiment_label"] == "Negative").mean()
        category_diversity = g["category"].nunique()
        photo_ratio = g["has_photo"].fillna(False).astype(bool).mean() if "has_photo" in g else 0.0

        valid_times = g["parsed_time"].dropna()
        if len(valid_times) >= 2:
            span_hours = max((valid_times.max() - valid_times.min()).total_seconds() / 3600.0, 0.5)
        else:
            span_hours = 1.0
        report_velocity = report_count / span_hours

        rows.append({
            "area": area,
            "report_count": report_count,
            "negative_ratio": round(float(negative_ratio), 3),
            "category_diversity": category_diversity,
            "photo_ratio": round(float(photo_ratio), 3),
            "report_velocity_per_hr": round(float(report_velocity), 3),
        })

    return pd.DataFrame(rows)


def detect_anomalies(unified_df: pd.DataFrame, contamination: float = 0.15) -> dict:
    """
    Runs IsolationForest over per-area features.
    Returns a ranked list of areas with an anomaly_score (higher = more unusual)
    and a boolean is_anomaly flag.
    """
    features_df = build_area_features(unified_df)
    if len(features_df) < 3:
        # Not enough distinct areas to fit a meaningful model.
        features_df["anomaly_score"] = 0.0
        features_df["is_anomaly"] = False
        return {
            "areas": features_df.sort_values("report_count", ascending=False).to_dict(orient="records"),
            "note": "Fewer than 3 distinct areas reported so far — anomaly ranking is not yet statistically meaningful. Showing raw counts instead.",
        }

    feature_cols = ["report_count", "negative_ratio", "category_diversity",
                     "photo_ratio", "report_velocity_per_hr"]
    X = features_df[feature_cols].to_numpy(dtype=float)

    # guard against degenerate contamination values for very small N
    n = len(features_df)
    contamination = float(np.clip(contamination, 1.0 / n, 0.5))

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
    )
    model.fit(X)

    raw_scores = model.decision_function(X)   # higher = more normal
    preds = model.predict(X)                  # -1 anomaly, 1 normal

    # Flip + rescale so higher = more anomalous, roughly 0..1
    anomaly_score = (raw_scores.max() - raw_scores) / (raw_scores.max() - raw_scores.min() + 1e-9)

    features_df["anomaly_score"] = np.round(anomaly_score, 3)
    features_df["is_anomaly"] = preds == -1

    ranked = features_df.sort_values("anomaly_score", ascending=False)
    return {
        "areas": ranked.to_dict(orient="records"),
        "note": None,
    }
