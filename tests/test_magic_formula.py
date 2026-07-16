"""마법공식(Magic Formula) 이익수익률(EY)·투하자본수익률(ROC) 크로스섹션 노출 (TDD).

배경: 조엘 그린블라트 마법공식의 두 팩터를 크로스섹션(get_cross_section/metrics_at)에 노출해
산점도/스크리닝에 쓸 수 있게 한다(GPA 노출과 동일 패턴).

계산식(사용자 제공 스크린샷 기준):
  EBIT = 당기순이익 + 법인세비용 + 이자비용                     … 손익계산서 항목(TTM 4분기 합)
  여유자금(excess cash):
     wcn = max(0, 유동부채 - 유동자산 + 현금)                   … 운전자본 소요 현금
     excess_cash_final = 현금 - wcn
  기업가치 EV = 시가총액 + 총부채 - excess_cash_final           … 재무상태표 항목은 시점 스냅샷
  이익수익률 EY = EBIT / EV (%)
  투하자본 IC = (유동자산 - 유동부채) + (비유동자산 - 감가상각비)  … 재무상태표 시점 스냅샷
  투하자본수익률 ROC = EBIT / IC (%)

TTM 대상은 손익계산서 항목(당기순이익/법인세비용/이자비용)뿐이다. 재무상태표(대차대조표)
항목(유동자산/유동부채/비유동자산/총부채/현금/감가상각비)은 특정 시점 잔액이라 스냅샷(_fin)이다.
"""
from __future__ import annotations

import sqlite3

import pytest

from src.backtest.data_access import METRIC_FIELD_DESCRIPTIONS, metrics_at
from src.db import init_db
from src.version import shift_quarter as _shift_quarter

_Q = "2026Q1"
_DISCLOSED = "2026-05-15"
_ASOF = "2026-06-30"

# 손익계산서 항목(TTM: 4분기 합) vs 재무상태표 항목(스냅샷: 최신 분기 1건)
_TTM_KEYS = {"net_income", "tax_expense", "interest_expense"}
_SNAPSHOT_KEYS = {
    "current_assets", "current_liabilities", "non_current_assets",
    "total_liabilities", "cash", "depreciation", "total_equity",
}


def _seed(tmp_path, name, *, market_cap, ttm_per_quarter: dict, snapshot: dict) -> sqlite3.Connection:
    """단일 종목을 시드한다. ttm_per_quarter는 분기당 값(4분기 반복 시드 → TTM=4×값),
    snapshot은 최신 분기 1건만 시드(시점 잔액)."""
    db = tmp_path / f"{name}.db"
    init_db(str(db))
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO company(stock_code, name, market, sector) VALUES (?,?,?,?)",
        ("005930", "삼성전자", "KOSPI", "반도체"),
    )
    for key, per_q in ttm_per_quarter.items():
        assert key in _TTM_KEYS
        for i in range(4):
            conn.execute(
                "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
                "VALUES (?,?,?,?,?)",
                ("005930", _shift_quarter(_Q, -i), _DISCLOSED, key, float(per_q)),
            )
    for key, val in snapshot.items():
        assert key in _SNAPSHOT_KEYS
        conn.execute(
            "INSERT INTO financials(stock_code, quarter, disclosed_date, account_key, amount) "
            "VALUES (?,?,?,?,?)",
            ("005930", _Q, _DISCLOSED, key, float(val)),
        )
    conn.execute(
        "INSERT INTO prices(stock_code, date, close, market_cap) VALUES (?,?,?,?)",
        ("005930", _ASOF, 72000.0, float(market_cap)),
    )
    conn.commit()
    return conn


def test_metrics_at_exposes_earnings_yield_and_roc_normal_case(tmp_path):
    """정상 케이스 — 손으로 계산한 EY/ROC와 일치."""
    conn = _seed(
        tmp_path, "magic_ok",
        market_cap=20_000.0,
        ttm_per_quarter={"net_income": 250.0, "tax_expense": 50.0, "interest_expense": 25.0},
        snapshot={
            "current_assets": 5_000.0, "current_liabilities": 3_000.0,
            "non_current_assets": 8_000.0, "total_liabilities": 6_000.0,
            "cash": 1_000.0, "depreciation": 500.0, "total_equity": 10_000.0,
        },
    )
    r = metrics_at(conn, _ASOF)[0]
    # EBIT(TTM) = (250+50+25)*4 = 1300
    # excess_cash_raw = 3000-5000+1000 = -1000 → wcn=0 → excess_cash_final = 1000
    # EV = 20000 + 6000 - 1000 = 25000 → EY = 1300/25000*100 = 5.2
    assert r["earnings_yield"] == pytest.approx(1300.0 / 25_000.0 * 100)
    # IC = (5000-3000)+(8000-500) = 9500 → ROC = 1300/9500*100
    assert r["roc"] == pytest.approx(1300.0 / 9_500.0 * 100)
    assert r["roc_estimated"] is False  # 감가상각비 실측값 사용 → 근사치 아님
    conn.close()


def test_roc_approximates_depreciation_as_zero_when_missing(tmp_path):
    """감가상각비가 없는 종목(예: 삼성전자 — DART 표준 API에 계정 자체가 없음)은
    ROC를 아예 None으로 두지 않고, 감가상각비=0으로 근사해 IC를 계산한다(비유동자산을
    깎지 않아 IC가 실제보다 커지므로 ROC는 실제보다 낮게=보수적으로 나오는 안전한 근사).
    이 경우 roc_estimated=True로 근사치임을 표시한다."""
    conn = _seed(
        tmp_path, "magic_no_dep",
        market_cap=20_000.0,
        ttm_per_quarter={"net_income": 250.0, "tax_expense": 50.0, "interest_expense": 25.0},
        snapshot={
            "current_assets": 5_000.0, "current_liabilities": 3_000.0,
            "non_current_assets": 8_000.0, "total_liabilities": 6_000.0,
            "cash": 1_000.0,  # depreciation 생략 — 데이터 없음
        },
    )
    # dep=0 근사 → IC = (5000-3000)+(8000-0) = 10000 → ROC = 1300/10000*100
    r = metrics_at(conn, _ASOF)[0]
    assert r["roc"] == pytest.approx(1300.0 / 10_000.0 * 100)
    assert r["roc_estimated"] is True
    conn.close()


def test_earnings_yield_none_when_enterprise_value_not_positive(tmp_path):
    """EV<=0이면 EY는 None(0/음수로 나누기 방지). 같은 행의 ROC는 정상 계산됨(가드 특정성)."""
    conn = _seed(
        tmp_path, "magic_ev0",
        market_cap=1_000.0,
        ttm_per_quarter={"net_income": 250.0, "tax_expense": 50.0, "interest_expense": 25.0},
        snapshot={
            "current_assets": 100_000.0, "current_liabilities": 0.0,
            "non_current_assets": 1_000.0, "total_liabilities": 0.0,
            "cash": 100_000.0, "depreciation": 0.0,
        },
    )
    # excess_cash_raw = 0-100000+100000 = 0 → wcn=0 → excess_cash_final = 100000
    # EV = 1000 + 0 - 100000 = -99000 <= 0 → EY None
    # IC = (100000-0)+(1000-0) = 101000 > 0 → ROC 계산됨
    r = metrics_at(conn, _ASOF)[0]
    assert r["earnings_yield"] is None
    assert r["roc"] is not None
    conn.close()


def test_roc_none_when_invested_capital_not_positive(tmp_path):
    """IC<=0이면 ROC는 None. 같은 행의 EY는 정상 계산됨(가드 특정성)."""
    conn = _seed(
        tmp_path, "magic_ic0",
        market_cap=10_000.0,
        ttm_per_quarter={"net_income": 250.0, "tax_expense": 50.0, "interest_expense": 25.0},
        snapshot={
            "current_assets": 1_000.0, "current_liabilities": 5_000.0,
            "non_current_assets": 1_000.0, "total_liabilities": 1_000.0,
            "cash": 500.0, "depreciation": 2_000.0,
        },
    )
    # IC = (1000-5000)+(1000-2000) = -5000 <= 0 → ROC None
    # excess_cash_raw = 5000-1000+500 = 4500 → wcn=4500 → excess_cash_final = 500-4500 = -4000
    # EV = 10000 + 1000 - (-4000) = 15000 > 0 → EY 계산됨
    r = metrics_at(conn, _ASOF)[0]
    assert r["roc"] is None
    assert r["roc_estimated"] is None  # roc 자체가 없으니 근사 여부도 해당 없음
    assert r["earnings_yield"] is not None
    conn.close()


def test_earnings_yield_roc_none_when_ebit_inputs_missing(tmp_path):
    """EBIT 입력(법인세비용 등)이 하나라도 없으면 EY/ROC 모두 None(추정 안 함)."""
    conn = _seed(
        tmp_path, "magic_no_tax",
        market_cap=20_000.0,
        ttm_per_quarter={"net_income": 250.0, "interest_expense": 25.0},  # tax_expense 누락
        snapshot={
            "current_assets": 5_000.0, "current_liabilities": 3_000.0,
            "non_current_assets": 8_000.0, "total_liabilities": 6_000.0,
            "cash": 1_000.0, "depreciation": 500.0,
        },
    )
    r = metrics_at(conn, _ASOF)[0]
    assert r["earnings_yield"] is None
    assert r["roc"] is None
    assert r["roc_estimated"] is None
    conn.close()


def test_earnings_yield_and_roc_in_field_descriptions():
    """스크리닝 노출의 단일 정의처(METRIC_FIELD_DESCRIPTIONS)에 EY/ROC가 있고,
    설명 문구에 한국어 용어(이익수익률/투하자본수익률)가 포함돼 LLM이 매핑할 수 있다."""
    assert "earnings_yield" in METRIC_FIELD_DESCRIPTIONS
    assert "roc" in METRIC_FIELD_DESCRIPTIONS
    assert "이익수익률" in METRIC_FIELD_DESCRIPTIONS["earnings_yield"]
    assert "투하자본수익률" in METRIC_FIELD_DESCRIPTIONS["roc"]
