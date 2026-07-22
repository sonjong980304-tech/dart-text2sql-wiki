"""올웨더 웹 노출 테스트 (AC11/AC14/AC15 지원).

웹 화면은 DB에 저장된 스냅샷을 읽기만 한다(조회 시 즉석 재계산 없음, AC11). chat.html/index.html
양쪽에 진입 버튼이 있어야 한다(AC15). fastapi 미설치 환경(셸 기본 python3)에서는 통째로 skip한다 —
웹 계층은 공유 venv에서만 구동되며, 순수 로직은 별도 테스트가 fastapi 없이 커버한다(기존 관례).
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

import web.app as webapp  # noqa: E402
from src.allweather.store import persist_snapshot  # noqa: E402
from src.config import CONFIG  # noqa: E402
from src.db import connect, init_db  # noqa: E402


def _seed(db: str) -> None:
    init_db(db)
    conn = connect(db)
    try:
        persist_snapshot(conn, {
            "computed_at": "2026-07-01",
            "weights": {"005930.KS": 0.3, "QQQ": 0.4, "TLT": 0.2, "411060.KS": 0.1},
            "cagr": 0.12, "mdd": -0.2, "sharpe": 1.1, "sortino": 1.4, "cumulative_return": 0.9,
            "backtest_curve": [{"date": "2016-01-31", "nav": 1.0}, {"date": "2026-07-01", "nav": 1.9}],
        })
    finally:
        conn.close()


def test_api_allweather_returns_stored_snapshot(tmp_path, monkeypatch):
    db = str(tmp_path / "w.db")
    _seed(db)
    monkeypatch.setattr(CONFIG, "db_path", db)
    r = TestClient(webapp.app).get("/api/allweather")
    assert r.status_code == 200
    d = r.json()
    assert d["available"] is True
    assert d["cagr"] == 0.12 and d["mdd"] == -0.2
    assert d["sharpe"] == 1.1 and d["cumulative_return"] == 0.9
    assert d["sortino"] == 1.4
    assert d["weights"]["005930.KS"] == 0.3
    assert len(d["backtest_curve"]) == 2


def test_api_allweather_does_not_recompute(tmp_path, monkeypatch):
    # AC11: 조회 시 walk-forward 재계산을 호출하지 않는다.
    db = str(tmp_path / "w2.db")
    _seed(db)
    monkeypatch.setattr(CONFIG, "db_path", db)
    import src.allweather.pipeline as pl

    called = []
    monkeypatch.setattr(pl, "run_walk_forward", lambda *a, **k: called.append(1))
    TestClient(webapp.app).get("/api/allweather")
    assert called == []


def test_api_allweather_empty_is_graceful(tmp_path, monkeypatch):
    db = str(tmp_path / "w3.db")
    init_db(db)
    monkeypatch.setattr(CONFIG, "db_path", db)
    r = TestClient(webapp.app).get("/api/allweather")
    assert r.status_code == 200
    assert r.json()["available"] is False


def test_allweather_page_served(tmp_path, monkeypatch):
    db = str(tmp_path / "w4.db")
    _seed(db)
    monkeypatch.setattr(CONFIG, "db_path", db)
    r = TestClient(webapp.app).get("/allweather")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "올웨더" in r.text or "allweather" in r.text.lower()


def test_chat_and_index_have_allweather_button():
    # AC15: chat.html + index.html 양쪽에 진입 링크(/allweather)가 있어야 한다.
    static = Path(webapp.__file__).resolve().parent / "static"
    chat = (static / "chat.html").read_text(encoding="utf-8")
    index = (static / "index.html").read_text(encoding="utf-8")
    assert "/allweather" in chat
    assert "/allweather" in index


def test_allweather_routes_not_shadowed():
    paths = {getattr(rt, "path", None) for rt in webapp.app.routes}
    assert "/api/allweather" in paths
    assert "/allweather" in paths
