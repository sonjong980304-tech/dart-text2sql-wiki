"""올웨더 스냅샷 이력 저장/조회 (all_weather_snapshot).

.omc/specs/brainstorming-all-weather-portfolio.md AC16 참고.

매달 계산 결과를 이력으로 계속 쌓는다(단일 최신값 덮어쓰기가 아니라 월별 행 추가 = INSERT).
AC13 텔레그램 델타 계산이 직전 달 저장값(직전 행)에 의존한다. weights/backtest_curve는 JSON으로
직렬화해 저장하고, 조회 시 파싱해 돌려준다. 테이블 스키마는 src/db.py SCHEMA_DDL에 정의돼 있다.
"""
from __future__ import annotations

import json

from ..version import now_iso


def persist_snapshot(conn, snapshot: dict) -> None:
    """스냅샷 dict를 all_weather_snapshot에 append(INSERT, UPDATE 아님 — 이력 추적)."""
    conn.execute(
        "INSERT INTO all_weather_snapshot("
        "computed_at, weights, cagr, mdd, sharpe, sortino, cumulative_return, backtest_curve, created_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (
            snapshot["computed_at"],
            json.dumps(snapshot["weights"], ensure_ascii=False),
            snapshot["cagr"],
            snapshot["mdd"],
            snapshot["sharpe"],
            snapshot["sortino"],
            snapshot["cumulative_return"],
            json.dumps(snapshot["backtest_curve"], ensure_ascii=False),
            now_iso(),
        ),
    )
    conn.commit()


def _row_to_snapshot(row) -> dict:
    """all_weather_snapshot 한 행(sqlite Row)을 JSON 파싱해 dict로 변환."""
    return {
        "id": row["id"],
        "computed_at": row["computed_at"],
        "weights": json.loads(row["weights"]) if row["weights"] else {},
        "cagr": row["cagr"],
        "mdd": row["mdd"],
        "sharpe": row["sharpe"],
        "sortino": row["sortino"],
        "cumulative_return": row["cumulative_return"],
        "backtest_curve": json.loads(row["backtest_curve"]) if row["backtest_curve"] else [],
        "created_at": row["created_at"],
    }


def get_latest_snapshots(conn, n: int = 2) -> list[dict]:
    """최근 n개 스냅샷을 최신순(id DESC)으로 반환한다(AC13 델타에 직전 2행 사용)."""
    rows = conn.execute(
        "SELECT id, computed_at, weights, cagr, mdd, sharpe, sortino, cumulative_return, "
        "backtest_curve, created_at FROM all_weather_snapshot ORDER BY id DESC LIMIT ?",
        (n,),
    ).fetchall()
    return [_row_to_snapshot(r) for r in rows]


def get_latest_snapshot(conn) -> dict | None:
    """가장 최근 스냅샷 1개. 이력이 없으면 None."""
    rows = get_latest_snapshots(conn, 1)
    return rows[0] if rows else None
