"""
sentiment.py
------------
Lightweight, dependency-free sentiment scoring for election-report text
(citizen issues, observer reports, and optionally social-media style text).

Deliberately avoids nltk/vaderSentiment/textblob corpus downloads so it works
fully offline. Uses a small hand-built lexicon tuned for civic/election
complaint language (queues, machine faults, intimidation, bribery, etc.)
plus simple negation + intensifier handling. Good enough for triage/trend
purposes -- swap in vaderSentiment or a transformer model for production use.
"""

from __future__ import annotations

import re
import pandas as pd

POSITIVE_WORDS = {
    "good": 1.5, "great": 2.0, "excellent": 2.5, "smooth": 1.8, "calm": 1.2,
    "fair": 1.5, "orderly": 1.5, "resolved": 1.8, "efficient": 1.6,
    "helpful": 1.5, "safe": 1.5, "peaceful": 1.8, "clean": 1.0,
    "quick": 1.2, "friendly": 1.4, "transparent": 1.6, "verified": 1.0,
    "working": 1.0, "operational": 1.0, "thank": 1.5, "thanks": 1.5,
    "appreciate": 1.6, "improved": 1.4,
}

NEGATIVE_WORDS = {
    "bribe": -3.0, "bribing": -3.0, "corrupt": -2.8, "corruption": -2.8,
    "fraud": -3.0, "rigged": -3.0, "rigging": -3.0, "intimidation": -2.6,
    "intimidated": -2.6, "threat": -2.4, "threatened": -2.4, "violence": -3.0,
    "violent": -2.8, "fight": -2.0, "clash": -2.2, "attack": -2.6,
    "missing": -1.6, "broken": -1.8, "error": -1.4, "fault": -1.6,
    "faulty": -1.8, "delay": -1.2, "delayed": -1.2, "long": -0.8,
    "queue": -0.6, "queues": -0.6, "crowded": -1.2, "chaos": -2.2,
    "chaotic": -2.2, "unsafe": -2.0, "problem": -1.2, "issue": -0.8,
    "malfunction": -2.0, "tampered": -2.8, "tampering": -2.8,
    "harassment": -2.6, "harassed": -2.6, "denied": -1.8, "blocked": -1.6,
    "illegal": -2.4, "unfair": -1.8, "slow": -1.0, "confusion": -1.4,
    "confused": -1.2, "shortage": -1.6, "understaffed": -1.6,
    "misconduct": -2.4, "abuse": -2.6, "abused": -2.6, "irregularity": -2.0,
    "irregularities": -2.0, "suspicious": -1.8, "power outage": -1.6,
    "no staff": -1.8, "absent": -1.4,
}

NEGATIONS = {"not", "no", "never", "without", "n't", "none"}
INTENSIFIERS = {"very": 1.5, "extremely": 2.0, "highly": 1.5, "really": 1.3, "so": 1.2}

_word_re = re.compile(r"[a-zA-Z']+")


def score_text(text: str) -> dict:
    """
    Score a single piece of text.
    Returns {"compound": float, "label": "Positive"|"Negative"|"Neutral", "hits": [...]}
    compound is roughly in [-1, 1] after normalization.
    """
    if not text or not isinstance(text, str):
        return {"compound": 0.0, "label": "Neutral", "hits": []}

    tokens = _word_re.findall(text.lower())
    raw_score = 0.0
    hits = []

    for i, tok in enumerate(tokens):
        weight = POSITIVE_WORDS.get(tok, NEGATIVE_WORDS.get(tok))
        if weight is None:
            continue

        window = tokens[max(0, i - 3):i]
        negated = any(w in NEGATIONS for w in window)
        intensity = 1.0
        for w in window:
            if w in INTENSIFIERS:
                intensity = max(intensity, INTENSIFIERS[w])

        contribution = weight * intensity
        if negated:
            contribution *= -0.8

        raw_score += contribution
        hits.append({"token": tok, "weight": round(contribution, 2), "negated": negated})

    # normalize into [-1, 1] with a soft cap
    if raw_score == 0:
        compound = 0.0
    else:
        compound = max(-1.0, min(1.0, raw_score / (abs(raw_score) + 4)))

    if compound >= 0.15:
        label = "Positive"
    elif compound <= -0.15:
        label = "Negative"
    else:
        label = "Neutral"

    return {"compound": round(compound, 3), "label": label, "hits": hits}


def score_dataframe(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """Adds 'sentiment_compound' and 'sentiment_label' columns to a copy of df."""
    out = df.copy()
    scored = out[text_col].fillna("").apply(score_text)
    out["sentiment_compound"] = scored.apply(lambda s: s["compound"])
    out["sentiment_label"] = scored.apply(lambda s: s["label"])
    return out


def sentiment_summary(df: pd.DataFrame) -> dict:
    """Aggregate sentiment counts + average compound score, optionally by area."""
    if df.empty:
        return {"overall": {"Positive": 0, "Negative": 0, "Neutral": 0}, "avg_compound": 0.0, "by_area": {}}

    counts = df["sentiment_label"].value_counts().to_dict()
    overall = {k: int(counts.get(k, 0)) for k in ["Positive", "Negative", "Neutral"]}
    avg_compound = float(df["sentiment_compound"].mean())

    by_area = {}
    if "area" in df.columns:
        grouped = df.groupby("area")["sentiment_compound"].mean().round(3)
        by_area = grouped.to_dict()

    return {"overall": overall, "avg_compound": round(avg_compound, 3), "by_area": by_area}
