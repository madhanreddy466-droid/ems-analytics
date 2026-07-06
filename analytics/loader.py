"""
loader.py
---------
Converts the JSON shapes produced by the EMS front-end (the "Export JSON"
buttons in the Admin panel, or the raw localStorage arrays) into pandas
DataFrames that the rest of the analytics package can work with.

Expected shapes (matching ems-issues / ems-observer-reports / ems-analyst-reports
in the front-end's localStorage):

issues: [
  {"id": 173..., "time": "7/1/2026, 10:04:12 AM", "user": "APZ1234567",
   "issue": "Long queue at booth", "status": "Pending", "photo": "data:..."}
]

observer_reports: [
  {"id": 173..., "time": "...", "observer": "APZ1234567",
   "area": "Booth #2034, District A", "problem": "Machine error",
   "photos": ["data:...", ...], "status": "Submitted"}
]

analyst_reports: [
  {"id": 173..., "time": "...", "analyst": "APZ1234567", "notes": "...",
   "summary": "...", "categories": {...}, "turnoutSeries": [...]}
]
"""

from __future__ import annotations

import pandas as pd


ISSUE_COLUMNS = ["id", "time", "user", "issue", "status", "photo"]
OBSERVER_COLUMNS = ["id", "time", "observer", "area", "problem", "photos", "status"]
ANALYST_COLUMNS = ["id", "time", "analyst", "notes", "summary", "categories", "turnoutSeries"]


def _to_df(records: list, columns: list[str]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(records)
    for col in columns:
        if col not in df.columns:
            df[col] = None
    df = df[columns]
    if "time" in df.columns:
        df["parsed_time"] = pd.to_datetime(df["time"], errors="coerce", format="mixed")
    return df


def issues_to_df(issues: list) -> pd.DataFrame:
    """Citizen-submitted issues -> DataFrame with a text column for NLP work."""
    df = _to_df(issues, ISSUE_COLUMNS)
    df["has_photo"] = df["photo"].fillna("").astype(str).str.len() > 0
    df["text"] = df["issue"].fillna("")
    df["source"] = "citizen"
    return df


def observer_reports_to_df(reports: list) -> pd.DataFrame:
    """Observer-submitted reports -> DataFrame with a text column for NLP work."""
    df = _to_df(reports, OBSERVER_COLUMNS)
    df["photo_count"] = df["photos"].apply(lambda p: len(p) if isinstance(p, list) else 0)
    df["text"] = df["problem"].fillna("")
    df["source"] = "observer"
    return df


def analyst_reports_to_df(reports: list) -> pd.DataFrame:
    df = _to_df(reports, ANALYST_COLUMNS)
    return df


def unify_reports(issues_df: pd.DataFrame, observer_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine citizen issues + observer reports into one long table with common
    columns (id, time, parsed_time, area/user, text, status, source).
    Used as the input to sentiment analysis and anomaly detection.
    """
    left = issues_df.rename(columns={"user": "actor"})[
        ["id", "time", "parsed_time", "actor", "text", "status", "source", "has_photo"]
    ].copy()
    left["area"] = None

    right = observer_df.rename(columns={"observer": "actor"})[
        ["id", "time", "parsed_time", "actor", "area", "text", "status", "source", "photo_count"]
    ].copy()
    right["has_photo"] = right["photo_count"] > 0
    right = right.drop(columns=["photo_count"])

    combined = pd.concat([left, right], ignore_index=True, sort=False)
    combined["area"] = combined["area"].fillna("Unspecified")
    return combined


def load_from_payload(payload: dict) -> dict:
    """
    payload: {"issues": [...], "observer_reports": [...], "analyst_reports": [...]}
    Returns a dict of DataFrames plus a unified report table.
    """
    issues_df = issues_to_df(payload.get("issues", []))
    observer_df = observer_reports_to_df(payload.get("observer_reports", []))
    analyst_df = analyst_reports_to_df(payload.get("analyst_reports", []))
    unified_df = unify_reports(issues_df, observer_df)
    return {
        "issues": issues_df,
        "observer_reports": observer_df,
        "analyst_reports": analyst_df,
        "unified": unified_df,
    }
