"""Opt-in read-only smoke check of the demo routes against the configured DB.

Mounts only auth, email, and Gmail routes, so the assistant stack and background
indexer never start during verification.
"""

import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.api.routes.auth import router as auth_router  # noqa: E402
from app.api.routes.emails import router as emails_router  # noqa: E402
from app.api.routes.gmail import router as gmail_router  # noqa: E402


def main() -> None:
    app = FastAPI()
    app.include_router(auth_router)
    app.include_router(emails_router)
    app.include_router(gmail_router)
    client = TestClient(app)
    login = client.post("/api/auth/demo")
    assert login.status_code == 200, login.text
    profile = client.get("/api/auth/me")
    assert profile.status_code == 200, profile.text
    assert profile.json()["email"] == "demo@gmail.com"
    assert profile.json()["is_demo"] is True

    emails = client.get("/api/emails", params={"range": "all_time", "per_page": 50})
    assert emails.status_code == 200, emails.text
    assert emails.json()["total"] >= 30
    categories = {item["gmail_category"] for item in emails.json()["emails"]}
    assert categories == {"Primary", "Social", "Promotions", "Updates", "Forums"}
    assert all(item["ai_label"] for item in emails.json()["emails"])
    charts = client.get("/api/analytics/charts", params={"range": "all_time"})
    assert charts.status_code == 200, charts.text
    assert {item["name"] for item in charts.json()["categories"]} == categories
    detail = client.get(f"/api/emails/{emails.json()['emails'][0]['id']}")
    assert detail.status_code == 200, detail.text
    assert detail.json()["body"]

    labels = client.get("/api/labels")
    assert labels.status_code == 200, labels.text
    assert any(item["type"] == "ai" for item in labels.json()["labels"])
    assert any(item["type"] == "user" for item in labels.json()["labels"])
    assert client.post("/api/gmail/sync").status_code == 403
    print({"demo_login": "ok", "messages": emails.json()["total"],
           "gmail_categories": sorted(categories), "gmail_sync_blocked": True})


if __name__ == "__main__":
    main()
