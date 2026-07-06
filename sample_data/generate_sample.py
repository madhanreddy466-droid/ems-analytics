"""
generate_sample.py
-------------------
Generates sample_issues.json and sample_observer_reports.json in the exact
shape the EMS front-end exports (Admin panel -> Export JSON), so you can
try the backend without wiring up the real page first:

    python sample_data/generate_sample.py
    curl -X POST http://localhost:5000/api/data/upload \
         -H "Content-Type: application/json" \
         -d @sample_data/combined_payload.json
"""

import json
import random
from datetime import datetime, timedelta

random.seed(7)

AREAS = [
    "Booth #1245, District North", "Booth #2034, District A",
    "Booth #789, Region West", "Booth #456, Region East",
    "Booth #3311, District South",
]

CITIZEN_ISSUES = [
    "Long queue at the booth, waited over an hour",
    "EVM machine showing an error message",
    "No staff present at the counter this morning",
    "Everything was smooth and well organized, thank you",
    "Suspicious individuals near the entrance, felt intimidated",
    "Power outage caused delays for 30 minutes",
    "Booth was clean and volunteers were very helpful",
    "Heard about bribing near the parking area, unconfirmed",
]

OBSERVER_PROBLEMS = [
    "Machine malfunction, had to restart twice",
    "Crowd control issues, no distancing maintained",
    "Clash between two groups of supporters outside",
    "All operations running smoothly, no concerns",
    "Missing staff member at registration desk",
    "Reports of intimidation near the queue",
    "Technical network issue delayed verification",
    "Well organized, positive turnout so far",
]


def _rand_time(base_day, hour_range=(7, 19)):
    hour = random.randint(*hour_range)
    minute = random.randint(0, 59)
    dt = base_day.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return dt.strftime("%-m/%-d/%Y, %-I:%M:%S %p") if hasattr(dt, "strftime") else str(dt)


def generate_issues(n=40, base_day=None):
    base_day = base_day or datetime.now()
    issues = []
    for i in range(n):
        issues.append({
            "id": int((base_day + timedelta(seconds=i)).timestamp() * 1000),
            "time": _rand_time(base_day),
            "user": f"APZ{1000000 + random.randint(0, 999999):07d}",
            "issue": random.choice(CITIZEN_ISSUES),
            "status": random.choice(["Pending", "Pending", "Solved"]),
            "photo": "",
        })
    return issues


def generate_observer_reports(n=30, base_day=None):
    base_day = base_day or datetime.now()
    reports = []
    for i in range(n):
        reports.append({
            "id": int((base_day + timedelta(seconds=i + 100000)).timestamp() * 1000),
            "time": _rand_time(base_day),
            "observer": f"OBS{2000000 + random.randint(0, 999999):07d}",
            "area": random.choice(AREAS),
            "problem": random.choice(OBSERVER_PROBLEMS),
            "photos": [],
            "status": random.choice(["Submitted", "Submitted", "Resolved"]),
        })
    return reports


if __name__ == "__main__":
    issues = generate_issues()
    observer_reports = generate_observer_reports()

    with open("sample_data/sample_issues.json", "w") as f:
        json.dump(issues, f, indent=2)
    with open("sample_data/sample_observer_reports.json", "w") as f:
        json.dump(observer_reports, f, indent=2)
    with open("sample_data/combined_payload.json", "w") as f:
        json.dump({
            "issues": issues,
            "observer_reports": observer_reports,
            "analyst_reports": [],
        }, f, indent=2)

    print(f"Generated {len(issues)} issues and {len(observer_reports)} observer reports.")
