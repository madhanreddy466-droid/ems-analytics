# EMS Analytics Backend

Adds a Python/pandas/scikit-learn analytics layer to the existing EMS
(Election Monitoring System) HTML front-end, as a Flask API.

## What it adds

| Feature | Where | How |
|---|---|---|
| Historical/aggregate analysis | `analytics/loader.py` | pandas DataFrames built from citizen issues + observer reports |
| Turnout prediction | `analytics/turnout.py` | `RandomForestRegressor`, trained on a bootstrapped synthetic historical dataset (swap in real historical CSVs via `/api/analytics/turnout/train`) |
| Anomaly detection | `analytics/anomaly.py` | `IsolationForest` over per-booth features (report volume, negative-sentiment ratio, category diversity, photo-evidence ratio, report velocity) — flags booths worth an admin's attention |
| Sentiment analysis | `analytics/sentiment.py` | Dependency-free lexicon scorer tuned for election-complaint language (queues, EVM faults, intimidation, bribery, etc.), no network/corpus downloads needed |
| API layer | `app.py` | Flask, serves it all as JSON |

## Run it

```bash
pip install -r requirements.txt
python app.py
# -> http://localhost:5000
```

Try it immediately with generated sample data (no need to touch the HTML page first):

```bash
python sample_data/generate_sample.py
curl -X POST http://localhost:5000/api/data/upload \
     -H "Content-Type: application/json" \
     --data-binary @sample_data/combined_payload.json

curl http://localhost:5000/api/analytics/summary
curl http://localhost:5000/api/analytics/sentiment
curl http://localhost:5000/api/analytics/anomalies
curl http://localhost:5000/api/analytics/turnout/metrics
curl -X POST http://localhost:5000/api/analytics/turnout/predict \
     -H "Content-Type: application/json" \
     -d '{"records":[{"hour":11,"booth_type":"urban","weather":"clear","is_weekend":0,"prior_election_turnout":60}]}'
```

## Wiring it to the existing EMS HTML page

The Admin panel already stores three arrays in `localStorage`
(`ems-issues`, `ems-observer-reports`, `ems-analyst-reports`) and already
has "Export JSON" buttons that call `downloadJSON(...)`. The smallest
change to feed the backend is to add one function alongside those and call
it whenever you want fresh analytics — e.g. add a button next to "Refresh"
in the Admin panel:

```html
<button class="btn" onclick="syncToAnalyticsBackend()">Sync to Analytics</button>
```

```js
const ANALYTICS_API = "http://localhost:5000";

async function syncToAnalyticsBackend(){
  const payload = {
    issues: loadIssuesFromStorage(),
    observer_reports: loadObserverReports(),
    analyst_reports: loadAnalystReports()
  };
  try{
    await fetch(`${ANALYTICS_API}/api/data/upload`, {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify(payload)
    });
    showPopup("Synced","Data sent to analytics backend.");
    loadAnalyticsPanels(); // see below
  }catch(e){
    showPopup("Sync Failed","Could not reach the analytics backend.");
  }
}

async function loadAnalyticsPanels(){
  const [summary, sentimentData, anomalies] = await Promise.all([
    fetch(`${ANALYTICS_API}/api/analytics/summary`).then(r=>r.json()),
    fetch(`${ANALYTICS_API}/api/analytics/sentiment`).then(r=>r.json()),
    fetch(`${ANALYTICS_API}/api/analytics/anomalies`).then(r=>r.json())
  ]);
  // render summary / sentimentData / anomalies into new cards in the
  // Analyst section, the same way renderAnalystCharts() already renders
  // ApexCharts from computeChartsData() — just swap the data source.
}
```

This keeps the front end's existing local-first UX (nothing breaks if the
backend is offline) while adding a clear "Sync to Analytics" action for
real pandas/scikit-learn results instead of the client-side keyword
heuristics currently used in `computeChartsData()` / `inferCategoryFromText()`.

## Notes & honest limitations

- **Turnout model**: same-day localStorage data has no real historical
  turnout curves to learn from, so the model trains on a bootstrapped
  synthetic dataset shaped like typical turnout patterns (morning/evening
  rush, booth type, weather). Retrain on real historical data via
  `POST /api/analytics/turnout/train` as soon as you have some — the
  synthetic version is for demonstrating the pipeline, not for real
  election forecasting.
- **Anomaly detection** ranks booths by *unusual reporting patterns*
  (volume, sentiment, category diversity, velocity) to help admins
  prioritize review. It does not and cannot determine that fraud
  occurred — a high anomaly score just means "this booth's report
  pattern looks different from the rest; look here first."
- **Sentiment analysis** is a small hand-built lexicon so it works fully
  offline with no corpus downloads. It's fine for triage/trend spotting;
  swap in `vaderSentiment` or a transformer model if you need
  research-grade accuracy and have network access to install/download one.
- In-memory `STORE` in `app.py` is for demo purposes — replace with a real
  database (Postgres, SQLite, etc.) before deploying this anywhere real
  users depend on it.
