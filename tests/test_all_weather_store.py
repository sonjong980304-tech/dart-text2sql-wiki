"""올웨더 스냅샷 이력 테이블 테스트 (AC16 + 스키마).

신규 all_weather_snapshot 테이블은 매달 계산 결과를 이력으로 계속 쌓는다(단일 최신값 덮어쓰기가
아니라 월별 행 추가). AC13 델타 계산이 이 이력(직전 행)에 의존한다. 기존 macro 테이블과 동일하게
자연어 SQL 질의 대상(QUERYABLE_TABLES)에는 포함하지 않는다.
"""
from __future__ import annotations

import sqlite3

from src.allweather.store import get_latest_snapshot, get_latest_snapshots, persist_snapshot
from src.db import QUERYABLE_TABLES, connect, init_db, schema_catalog


def _snap(computed_at: str, sam_w: float):
    return {
        "computed_at": computed_at,
        "weights": {"005930.KS": sam_w, "QQQ": 0.4, "TLT": 0.2, "411060.KS": 0.4 - sam_w},
        "cagr": 0.11,
        "mdd": -0.25,
        "sharpe": 1.05,
        "cumulative_return": 0.8,
        "backtest_curve": [{"date": "2016-01-31", "nav": 1.0}, {"date": computed_at, "nav": 1.8}],
    }


def test_table_created_by_init_db(tmp_path):
    db = str(tmp_path / "s.db")
    init_db(db)
    conn = sqlite3.connect(db)
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "all_weather_snapshot" in tables
        cols = {r[1] for r in conn.execute("PRAGMA table_info(all_weather_snapshot)")}
        assert {
            "computed_at", "weights", "cagr", "mdd", "sharpe",
            "cumulative_return", "backtest_curve", "created_at",
        } <= cols
    finally:
        conn.close()


def test_persist_appends_history_not_overwrite(tmp_path):
    # AC16: 두 번 저장하면 1행이 아니라 2행(이력).
    db = str(tmp_path / "s2.db")
    init_db(db)
    conn = connect(db)
    try:
        persist_snapshot(conn, _snap("2026-06-01", 0.25))
        persist_snapshot(conn, _snap("2026-07-01", 0.30))
        n = conn.execute("SELECT COUNT(*) FROM all_weather_snapshot").fetchone()[0]
        assert n == 2
    finally:
        conn.close()


def test_get_latest_snapshot_parses_json(tmp_path):
    db = str(tmp_path / "s3.db")
    init_db(db)
    conn = connect(db)
    try:
        persist_snapshot(conn, _snap("2026-06-01", 0.25))
        persist_snapshot(conn, _snap("2026-07-01", 0.30))
        latest = get_latest_snapshot(conn)
        assert latest["computed_at"] == "2026-07-01"
        assert latest["weights"]["005930.KS"] == 0.30  # JSON 파싱됨
        assert isinstance(latest["backtest_curve"], list)
        two = get_latest_snapshots(conn, 2)
        assert [s["computed_at"] for s in two] == ["2026-07-01", "2026-06-01"]  # 최신순
    finally:
        conn.close()


def test_get_latest_snapshot_empty_returns_none(tmp_path):
    db = str(tmp_path / "s4.db")
    init_db(db)
    conn = connect(db)
    try:
        assert get_latest_snapshot(conn) is None
    finally:
        conn.close()


def test_snapshot_table_not_queryable(tmp_path):
    # 기존 macro/metrics 테이블과 동일 관례 — 자연어 SQL 질의 대상 제외.
    assert "all_weather_snapshot" not in QUERYABLE_TABLES
    catalog = schema_catalog(str(tmp_path / "unused.db"))
    assert "all_weather_snapshot" not in catalog
