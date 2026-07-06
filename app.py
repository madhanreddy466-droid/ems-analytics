"""
app.py
------
Flask backend for the EMS (Election Monitoring System) front-end.

Run:
    pip install -r requirements.txt
    python app.py
    -> serves on http://localhost:5000

Data flow with the existing HTML/JS front end:
    The Admin panel already has "Export JSON" buttons for citizen issues,
    observer reports, and analyst reports (they call downloadJSON()).
    Point those same three localStorage arrays at this backend instead of/
    in addition to downloading files: POST them to /api/data/upload as
        {"issues": [...], "observer_reports": [...], "analyst_reports": [...]}
    Everything below then reads from the in-memory store that upload fills.
    (Swap the in-memory store for a real DB for production use.)

Endpoints:
    GET  /api/health
    POST /api/data/upload                      -> load a data snapshot
    GET  /api/analytics/summary                 -> pandas overview stats
    GET  /api/analytics/sentiment                -> sentiment breakdown
    GET  /api/analytics/anomalies                -> IsolationForest booth ranking
    GET  /api/analytics/turnout/metrics          -> current turnout model info
    POST /api/analytics/turnout/predict          -> predict turnout for booths
    POST /api/analytics/turnout/train            -> retrain on real historical CSV/JSON
"""

from __future__ import annotations

from flask import Flask, request, jsonify

from analytics import loader, sentiment, anomaly, turnout

app = Flask(__name__)

# ---- simple in-memory data store (swap for a DB in production) ----
STORE = {
    "issues": [],
    "observer_reports": [],
    "analyst_reports": [],
}


@app.after_request
def add_cors_headers(resp):
    # Manual CORS so the EMS HTML page (served from file:// or another port)
    # can call this API directly from the browser.
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return resp


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/data/upload", methods=["POST", "OPTIONS"])
def upload_data():
    if request.method == "OPTIONS":
        return "", 204

    payload = request.get_json(force=True, silent=True) or {}
    STORE["issues"] = payload.get("issues", STORE["issues"])
    STORE["observer_reports"] = payload.get("observer_reports", STORE["observer_reports"])
    STORE["analyst_reports"] = payload.get("analyst_reports", STORE["analyst_reports"])

    return jsonify({
        "status": "loaded",
        "counts": {
            "issues": len(STORE["issues"]),
            "observer_reports": len(STORE["observer_reports"]),
            "analyst_reports": len(STORE["analyst_reports"]),
        },
    })


def _current_frames():
    return loader.load_from_payload(STORE)


@app.route("/api/analytics/summary", methods=["GET"])
def analytics_summary():
    frames = _current_frames()
    unified = frames["unified"]

    if unified.empty:
        return jsonify({
            "total_reports": 0,
            "by_status": {},
            "by_source": {},
            "reports_with_photo_pct": 0.0,
            "distinct_areas": 0,
        })

    by_status = unified["status"].fillna("Unknown").value_counts().to_dict()
    by_source = unified["source"].value_counts().to_dict()
    photo_pct = round(float(unified["has_photo"].fillna(False).astype(bool).mean()) * 100, 1)
    distinct_areas = unified["area"].nunique()

    return jsonify({
        "total_reports": int(len(unified)),
        "by_status": by_status,
        "by_source": by_source,
        "reports_with_photo_pct": photo_pct,
        "distinct_areas": int(distinct_areas),
    })


@app.route("/api/analytics/sentiment", methods=["GET"])
def analytics_sentiment():
    frames = _current_frames()
    unified = frames["unified"]
    scored = sentiment.score_dataframe(unified, text_col="text")
    summary = sentiment.sentiment_summary(scored)

    sample_cols = ["id", "time", "actor", "area", "text", "sentiment_label", "sentiment_compound"]
    sample = scored[sample_cols].sort_values("sentiment_compound").to_dict(orient="records") if not scored.empty else []

    return jsonify({
        "summary": summary,
        "most_negative_reports": sample[:10],
        "most_positive_reports": sample[-10:][::-1] if sample else [],
    })


@app.route("/api/analytics/anomalies", methods=["GET"])
def analytics_anomalies():
    frames = _current_frames()
    contamination = float(request.args.get("contamination", 0.15))
    result = anomaly.detect_anomalies(frames["unified"], contamination=contamination)
    return jsonify(result)


@app.route("/api/analytics/turnout/metrics", methods=["GET"])
def turnout_metrics():
    return jsonify(turnout.model_metrics())


@app.route("/api/analytics/turnout/predict", methods=["POST"])
def turnout_predict():
    payload = request.get_json(force=True, silent=True) or {}
    records = payload.get("records", [])
    if not records:
        return jsonify({"error": "Provide 'records': a list of {hour, booth_type, weather?, is_weekend?, prior_election_turnout?}"}), 400
    try:
        predictions = turnout.predict(records)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"predictions": predictions})


@app.route("/api/analytics/turnout/train", methods=["POST"])
def turnout_train():
    """
    Retrain on real historical data if you have it:
    body: {"rows": [{"hour":9,"booth_type":"urban","weather":"clear",
                      "is_weekend":0,"prior_election_turnout":62.5,
                      "cumulative_turnout_pct":18.4}, ...]}
    """
    import pandas as pd
    payload = request.get_json(force=True, silent=True) or {}
    rows = payload.get("rows", [])
    if len(rows) < 20:
        return jsonify({"error": "Provide at least 20 historical rows for a meaningful fit."}), 400
    df = pd.DataFrame(rows)
    metrics = turnout.train_from_dataframe(df)
    return jsonify({"status": "trained", "metrics": metrics})


if __name__ == "__main__":
    app.run(debug=True, port=5001)
