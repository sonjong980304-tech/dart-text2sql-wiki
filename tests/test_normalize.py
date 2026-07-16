"""normalize_account 매핑 테스트.

핵심 개선: DART account_id의 접두사 변형(`ifrs_` 구표기)을 표준(`ifrs-full_`)과
동일하게 취급해 _DART_ID_MAP으로 매핑한다. 기존 매핑(이름매칭, 비지배 차단)은 회귀 없이 유지.
"""
from __future__ import annotations

import pytest

from src.ingest.normalize import normalize_account


# ── 사이클 1: account_id 접두사 정규화 (ifrs_ ≡ ifrs-full_) ──────────────
@pytest.mark.parametrize(
    "account_id, expected",
    [
        ("ifrs_ProfitLossAttributableToOwnersOfParent", "controlling_net_income"),
        ("ifrs_EquityAttributableToOwnersOfParent", "controlling_equity"),
        ("ifrs_Revenue", "revenue"),
        ("ifrs_ProfitLoss", "net_income"),
        ("ifrs_Assets", "total_assets"),
        ("ifrs_Equity", "total_equity"),
    ],
)
def test_ifrs_prefix_variant_maps_like_full(account_id, expected):
    """`ifrs_X`(하이픈 없는 구표기)도 `ifrs-full_X`와 동일 표준키로 매핑된다."""
    # account_nm은 비어 있어도 account_id만으로 매핑돼야 한다(계정명 표기에 안 흔들림).
    assert normalize_account("", account_id) == expected


# ── 회귀: 기존 표준(ifrs-full_) 매핑 유지 ────────────────────────────────
@pytest.mark.parametrize(
    "account_id, expected",
    [
        ("ifrs-full_ProfitLossAttributableToOwnersOfParent", "controlling_net_income"),
        ("ifrs-full_EquityAttributableToOwnersOfParent", "controlling_equity"),
        ("ifrs-full_Revenue", "revenue"),
    ],
)
def test_full_prefix_still_maps(account_id, expected):
    assert normalize_account("아무이름", account_id) == expected


# ── 회귀: 이름 매칭 유지 (account_id 없을 때) ────────────────────────────
@pytest.mark.parametrize(
    "name, expected",
    [
        ("매출액", "revenue"),
        ("영업수익", "revenue"),
        ("판매비와관리비", "sga"),
        ("매출원가", "cost_of_sales"),
        ("당기순이익", "net_income"),
    ],
)
def test_name_matching_still_works(name, expected):
    assert normalize_account(name, None) == expected


# ── 회귀: 비지배지분은 여전히 차단(None) ────────────────────────────────
@pytest.mark.parametrize(
    "name, account_id",
    [
        ("비지배지분", "ifrs-full_ProfitLossAttributableToNoncontrollingInterests"),
        ("비지배지분", "ifrs_ProfitLossAttributableToNoncontrollingInterests"),
        ("비지배지분에귀속되는당기순이익(손실)", "ifrs-full_ProfitLossAttributableToNoncontrollingInterests"),
    ],
)
def test_noncontrolling_still_excluded(name, account_id):
    """'비지배'는 _EXCLUDE로 차단돼 controlling_* 로 오매칭되지 않는다."""
    assert normalize_account(name, account_id) is None


# ── 사이클 2: _DART_ID_MAP 확장 (이름매칭 계정을 표준 element ID로 승격) ──
@pytest.mark.parametrize(
    "account_id, expected",
    [
        ("ifrs-full_CostOfSales", "cost_of_sales"),
        ("ifrs-full_GrossProfit", "gross_profit"),
        ("ifrs-full_CurrentAssets", "current_assets"),
        ("ifrs-full_NoncurrentAssets", "non_current_assets"),
        ("ifrs-full_CurrentLiabilities", "current_liabilities"),
        ("ifrs-full_NoncurrentLiabilities", "non_current_liabilities"),
        ("ifrs-full_CashFlowsFromUsedInOperatingActivities", "operating_cashflow"),
        ("ifrs-full_DividendsPaid", "dividend"),
        # 접두사 변형(ifrs_)도 확장된 코드에 동일 적용
        ("ifrs_CurrentAssets", "current_assets"),
        ("ifrs_CostOfSales", "cost_of_sales"),
    ],
)
def test_expanded_id_map_promotes_name_only_accounts(account_id, expected):
    """표준 element ID만으로(계정명 없이) 매핑된다 — 비표준 계정명에 안 흔들림."""
    assert normalize_account("", account_id) == expected


@pytest.mark.parametrize(
    "account_id",
    [
        "ifrs-full_CashFlowsFromUsedInInvestingActivities",   # 투자활동 — 영업현금흐름 아님
        "ifrs-full_CashFlowsFromUsedInFinancingActivities",   # 재무활동 — 영업현금흐름 아님
    ],
)
def test_investing_financing_cashflow_not_mapped(account_id):
    """영업활동 외 현금흐름은 operating_cashflow로 오매칭되지 않는다(None)."""
    assert normalize_account("", account_id) is None


# ── 사이클 3: 마법공식(EY/ROC) 계산용 tax_expense / cash 정규화 규칙 추가 ──
@pytest.mark.parametrize(
    "name, expected",
    [
        ("법인세비용", "tax_expense"),
        ("현금및현금성자산", "cash"),
    ],
)
def test_tax_and_cash_name_matching(name, expected):
    """이름 매칭으로 법인세비용→tax_expense, 현금및현금성자산→cash."""
    assert normalize_account(name, None) == expected


@pytest.mark.parametrize(
    "account_id, expected",
    [
        ("ifrs-full_IncomeTaxExpenseContinuingOperations", "tax_expense"),
        ("ifrs-full_CashAndCashEquivalents", "cash"),
        # 접두사 변형(ifrs_)도 동일 표준키로 매핑
        ("ifrs_IncomeTaxExpenseContinuingOperations", "tax_expense"),
        ("ifrs_CashAndCashEquivalents", "cash"),
    ],
)
def test_tax_and_cash_id_matching(account_id, expected):
    """account_id 우선 매칭 — 계정명이 비표준이어도 표준 element ID로 안정 매핑."""
    assert normalize_account("아무이름", account_id) == expected


def test_income_tax_before_tax_not_mapped_to_tax_expense():
    """'법인세비용차감전순이익'은 tax_expense가 아니다('차감전' _EXCLUDE로 차단 → None).

    account_id(ifrs-full_ProfitLossBeforeTax)도 tax_expense로 매핑되지 않는다.
    """
    assert normalize_account("법인세비용차감전순이익", "ifrs-full_ProfitLossBeforeTax") is None
    assert normalize_account("법인세비용차감전순이익", None) is None


@pytest.mark.parametrize(
    "name, account_id",
    [
        ("기초현금및현금성자산", "dart_CashAndCashEquivalentsAtBeginningOfPeriodCf"),
        ("기말현금및현금성자산", "dart_CashAndCashEquivalentsAtEndOfPeriodCf"),
        ("분기말의현금및현금성자산", "dart_CashAndCashEquivalentsAtEndOfPeriodCf"),
    ],
)
def test_cashflow_begin_end_cash_not_mapped_to_cash(name, account_id):
    """현금흐름표 기초/기말 현금은 재무상태표 cash(현금및현금성자산 스냅샷)로 오매칭되지 않는다.

    이름에 '현금및현금성자산'을 부분문자열로 포함하지만 '기초'/'기말'이 _EXCLUDE로 차단되고,
    이 계정들의 account_id(dart_...Cf)는 _DART_ID_MAP에 없어 cash로 매핑되지 않는다.
    """
    assert normalize_account(name, account_id) is None


@pytest.mark.parametrize(
    "name",
    [
        # 실제 data/market.db raw_reports에서 확인된 현금흐름표 오염 변형들.
        "현금및현금성자산에 대한 환율변동효과",
        "외화표시 현금및현금성자산의 환율변동효과",
        "현금및현금성자산의순증가(감소)",
        "현금및현금성자산의 증감",
        "현금및현금성자산의증가(감소)",
        "보유현금및현금성자산환산효과",
        "환율변동효과 반영전 현금및현금성자산의 순증가(감소)",
    ],
)
def test_cashflow_change_and_fx_cash_not_mapped_to_cash(name):
    """현금흐름표의 환율변동효과/순증가(감소)/증감/환산효과 항목은 재무상태표 cash로 오매칭되지
    않는다 — '현금및현금성자산'을 부분문자열로 포함하지만 흐름표 수식어(환율변동/증가/감소/증감/
    환산효과)가 _EXCLUDE로 차단된다. 재무상태표 시점 잔액 하나만 cash로 잡아야 한다."""
    assert normalize_account(name, None) is None
