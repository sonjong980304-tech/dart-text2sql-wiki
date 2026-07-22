"""AC5: 백테스트 정확성 factcheck (US-7).

.omc/specs/brainstorming-factcheck-eval.md AC5 참고:
(a) 기존 백테스트 하드차단 pytest(생존편향/미래참조/공매도 가드) + NAV·베타 계산 검증을
    subprocess로 재실행해 통과 여부를 기록한다.
(b) 코스피 지수 최근 1년 실제 수익률 대비 동일가중 유니버스 백테스트 수익률을 신규
    시나리오(예: 다른 리밸런싱 주기)로 대조한다 — ±2.5%p(Round 12 ±2~3%p 범위 중 보수적으로
    채택). 비교는 새 로직을 만들지 않고 src/eval/factcheck/tolerance.py의
    within_pct_tolerance를 그대로 재사용한다.

원본 DB는 항상 src/eval/runner.py의 _isolated_copy/_cleanup_copy로 격리된 사본에서만
읽는다(기존 원본 보호 관례 재사용).
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from ...db import connect
from ..db_isolation import _cleanup_copy, _isolated_copy
from .tolerance import within_pct_tolerance

_REPO_ROOT = Path(__file__).resolve().parents[3]

# AC5(a) 대상: 하드차단 3종(생존편향/미래참조/공매도) 정적가드 + NAV·베타 계산 검증 선례.
# 레포 구조 변경에 대비해 실제 존재하는 파일만 대상으로 한다(이 시점 기준 3개 모두 존재 확인).
_PYTEST_TARGETS = [
    "tests/test_backtest_performance.py",
    "tests/test_backtest_weights.py",
    "tests/test_backtest_auditor_static_guard.py",
]

_INDEX_TOLERANCE_PCT = 0.025  # ±2.5%p

_KOSPI_PSEUDO_CODE = "KOSPI"  # prices 테이블에 코스피 지수를 담을 경우의 가상 종목코드(현재 미수집)

# "동일가중 유니버스"는 시가총액 필드 하나만 기준으로 두고 상한 n을 크게 잡아, 사실상
# 시가총액 데이터가 있는 종목 전부를 동일가중 편입하는 방식으로 근사한다. select_stocks는
# criteria가 빈 리스트면 무조건 빈 결과를 반환하므로(src/backtest/selection.py) criterion이
# 최소 1개는 있어야 한다.
_EQUAL_WEIGHT_CRITERIA = [{"key": "market_cap", "direction": "high"}]
_EQUAL_WEIGHT_N = 5000


def run_backtest_pytest_check() -> dict:
    """AC5(a): 백테스트 하드차단/검증 pytest를 subprocess로 재실행해 결과를 기록한다.

    반환: {"exit_code": int|None, "summary": str, "pass": bool, "note": str}
    """
    existing = [f for f in _PYTEST_TARGETS if (_REPO_ROOT / f).exists()]
    missing = [f for f in _PYTEST_TARGETS if f not in existing]
    note = f"파일 없음(스킵): {missing}" if missing else ""

    if not existing:
        return {"exit_code": None, "summary": "", "pass": False, "note": note or "대상 테스트 파일 없음"}

    result = subprocess.run(
        [sys.executable, "-m", "pytest", *existing, "-q"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
    )
    out_lines = [ln for ln in (result.stdout or "").strip().splitlines() if ln.strip()]
    err_lines = [ln for ln in (result.stderr or "").strip().splitlines() if ln.strip()]
    summary = out_lines[-1] if out_lines else (err_lines[-1] if err_lines else "")
    return {
        "exit_code": result.returncode,
        "summary": summary,
        "pass": result.returncode == 0,
        "note": note,
    }


def _fetch_kospi_1y_return(db_path: str) -> float:
    """격리 사본 DB에서 코스피 지수 최근 1년 실제 수익률(fraction)을 계산한다.

    prices 테이블의 stock_code='KOSPI'(scripts/backfill_kospi_index.py로 최초 적재,
    src/ingest/naver_prices.py가 개별종목과 동일한 방식으로 일일 갱신)의 최신 종가와,
    거기서 정확히 12개월 전 시점(date<=t 최근값, look-ahead 없음)의 종가를 비교한다.
    최신 종가와 데이터 전체의 첫 종가를 비교하면(예전 구현) 전체 이력(수년치) 누적
    수익률이 나와 "1년" 이름과 실제 계산이 어긋나므로 12개월 전 기준을 명시적으로 구한다.
    데이터가 없거나 12개월 전 시점 값이 아예 없으면(예: 백필 범위가 1년이 안 됨) ValueError를
    던져 호출부가 pass=None, note="측정불가"로 기록하게 한다(예외를 여기서 삼키지 않는다).
    """
    from ...backtest.data_access import _months_before

    conn = connect(db_path)
    try:
        latest = conn.execute(
            "SELECT date, close FROM prices WHERE stock_code = ? ORDER BY date DESC LIMIT 1",
            (_KOSPI_PSEUDO_CODE,),
        ).fetchone()
        if not latest:
            raise ValueError("prices 테이블에 코스피 지수 데이터가 없음(수집 파이프라인 미구축)")
        one_year_ago = _months_before(latest["date"], 12)
        base_row = conn.execute(
            "SELECT close FROM prices WHERE stock_code = ? AND date <= ? ORDER BY date DESC LIMIT 1",
            (_KOSPI_PSEUDO_CODE, one_year_ago),
        ).fetchone()
    finally:
        conn.close()
    first, last = (base_row["close"] if base_row else None), latest["close"]
    if not first:
        raise ValueError("코스피 지수 1년 전 시점 데이터가 없어 수익률을 계산할 수 없음")
    return (last / first) - 1


def _compute_equal_weight_return(db_path: str, scenario: dict) -> float:
    """시나리오 파라미터로 동일가중 유니버스 백테스트를 실행해 총수익률(fraction)을 반환한다.

    새 백테스트 로직을 만들지 않고 기존 src/backtest/primitives.py의
    run_backtest_primitive를 그대로 재사용한다(리밸런싱/수익률계산/생존편향 가드는 이미
    검증된 엔진에 위임).
    """
    from ...backtest.primitives import run_backtest_primitive

    conn = connect(db_path)
    try:
        result = run_backtest_primitive(
            conn,
            start_year=scenario["start_year"],
            end_year=scenario["end_year"],
            criteria=scenario.get("criteria", _EQUAL_WEIGHT_CRITERIA),
            n=scenario.get("n", _EQUAL_WEIGHT_N),
            rebalance=scenario.get("rebalance", "quarterly"),
            with_benchmark=False,
            market="KR",
        )
    finally:
        conn.close()
    return result["performance"]["total_return"] / 100.0


def run_backtest_index_check(scenarios: list[dict], db_path: str | None = None) -> list[dict]:
    """AC5(b): 시나리오별 동일가중 백테스트 수익률을 코스피 지수 최근 1년 실제 수익률과 대조한다.

    각 시나리오는 반드시 _isolated_copy로 격리된 DB 사본에서 실행하고, 실행 후(예외 발생
    여부와 무관하게) _cleanup_copy로 정리한다. 코스피 지수 데이터를 구할 수 없으면 예외를
    전파하지 않고 pass=None, note="측정불가"로 기록한다(이 경우 백테스트 계산 자체는
    수행하지 않는다).

    반환: [{"scenario":..., "expected":..., "actual":..., "pass": bool|None, "note": str}]
    """
    results = []
    for scenario in scenarios:
        name = scenario.get("name", str(scenario))
        copy_path = _isolated_copy(db_path)
        try:
            try:
                expected = _fetch_kospi_1y_return(copy_path)
            except Exception as exc:  # noqa: BLE001 — 지수 데이터 미확보는 실패가 아니라 "측정불가"
                results.append({
                    "scenario": name,
                    "expected": None,
                    "actual": None,
                    "pass": None,
                    "note": f"측정불가: {exc}",
                })
                continue
            actual = _compute_equal_weight_return(copy_path, scenario)
            results.append({
                "scenario": name,
                "expected": expected,
                "actual": actual,
                "pass": within_pct_tolerance(expected, actual, _INDEX_TOLERANCE_PCT),
                "note": "",
            })
        finally:
            _cleanup_copy(copy_path)
    return results
