"""총괄 차트 배선(src/agents/supervisor.py) 단위 테스트.

차트 종류 판정은 결정론적 패턴매칭 없이 전적으로 LLM(chart_agent.build_chart_freeform)에
위임한다 — wants_chart(question)이면 domain_results 모양과 무관하게 항상 chart_fallback_fn을
호출한다. (옛 _build_charts 결정론 경로가 숫자 필드 1개짜리 결과에서 같은 필드를 x·y에 둘 다
배정해 "pbr vs pbr 산점도"를 그리던 실사용 스크린샷 오판을 구조적으로 제거했다.)

검증 대상:
- wants_chart(question): 명시적 차트 키워드가 있을 때만 True(오탐 방지). 재시도 피드백이
  덧붙기 전 '원본 question'으로 판단해야 한다.
- answer_with_verification: wants_chart가 True면 (데이터 모양과 무관하게) chart_fallback_fn을
  호출해 결과를 chart_base64/chart_title/charts에 배선한다. 차트 판단 LLM은 chart_llm_fn(기본
  =llm_fn; web가 저가 role="chart"로 별도 주입)이 맡는다. 키워드 없으면/불확실 응답이면 차트 없음.
"""
from __future__ import annotations

from src.agents.supervisor import answer_with_verification, wants_chart


def _valid_verify(question, domain_results, llm_fn):
    return {"valid": True, "reason": "일치"}


# ── wants_chart — 결정론적 키워드 감지(오탐 방지) ────────────────────────────

def test_wants_chart_true_for_graph_keyword():
    assert wants_chart("삼성전자 최근 1년 주가 그래프 그려줘") is True


def test_wants_chart_true_for_chart_keyword():
    assert wants_chart("골든크로스 전략 백테스트 차트로 보여줘") is True


def test_wants_chart_true_for_english_plot():
    assert wants_chart("plot the nav curve") is True
    assert wants_chart("show me a chart") is True


def test_wants_chart_true_for_visualize_and_trend():
    assert wants_chart("스프레드 시각화 해줘") is True
    assert wants_chart("금리차 추이 보여줘") is True


def test_wants_chart_false_for_plain_question():
    """일반 질문(키워드 없음)은 False — 불필요한 차트 생성을 막는다(오탐 방지)."""
    assert wants_chart("삼성전자 PER 알려줘") is False
    assert wants_chart("지금 매크로 신호 어때?") is False
    assert wants_chart("PER 낮은 5개 회사") is False


def test_wants_chart_empty_is_false():
    assert wants_chart("") is False
    assert wants_chart(None) is False


# ── wants_chart면 데이터 모양과 무관하게 항상 chart_fallback_fn(LLM)로 위임 ──────
# 실사용 스크린샷 버그: "PBR 오름차순 나열해서 그래프" 요청에 옛 결정론 경로가 pbr 한 필드를
# x·y 둘 다에 배정해 "pbr vs pbr 산점도"를 그렸다. 이제 결정론 1단계를 제거해 그 오판 경로 자체가
# 없다 — scatter 모양이든 단일필드 스크리닝이든 무조건 LLM(chart_fallback_fn)이 그린다.

def test_always_uses_chart_fallback_even_for_scatter_shape():
    """옛 결정론 경로가 산점도로 렌더하던 scatter 모양(x/y/x_field/y_field, 심지어 x_field==
    y_field=='pbr')이어도, 이제는 반드시 chart_fallback_fn(LLM)에 맡긴다 — "pbr vs pbr" 오판 소거."""
    bt = {"result": {
        "x": [0.018, 0.4, 0.83], "y": [0.018, 0.4, 0.83], "labels": ["가", "나", "다"],
        "x_field": "pbr", "y_field": "pbr"}}

    def stub_route(question, llm_fn):
        return ["backtest"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"backtest": bt}

    called = {}

    def fake_chart_fallback(question, data, llm_fn):
        called["yes"] = True
        return {"chart_base64": "PNG", "chart_title": "LLM이 고른 차트"}

    res = answer_with_verification(
        "코스피 전체 종목 pbr 오름차순 나열해서 그래프 그려줘", conn=None, llm_fn="fake-llm",
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
        chart_fallback_fn=fake_chart_fallback,
    )
    assert called.get("yes") is True
    assert res["chart_base64"] == "PNG"
    assert res["chart_title"] == "LLM이 고른 차트"
    assert res["charts"] == [{"chart_base64": "PNG", "chart_title": "LLM이 고른 차트"}]


def test_always_uses_chart_fallback_for_single_field_pbr_screening():
    """PBR 스크린샷 시나리오 그대로: 숫자필드 1개(pbr)짜리 스크리닝 리스트도 LLM 폴백으로 간다."""
    rows = [
        {"stock_code": "005930", "name": "삼성전자", "pbr": 0.83},
        {"stock_code": "000660", "name": "SK하이닉스", "pbr": 1.2},
    ]

    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {"result": rows}}

    captured = {}

    def fake_chart_fallback(question, data, llm_fn):
        captured["data"] = data
        return {"chart_base64": "PNG", "chart_title": "정렬 막대"}

    res = answer_with_verification(
        "코스피 전체 종목 중 PBR 오름차순 나열해서 그래프 그려줘", conn=None, llm_fn="fake-llm",
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
        chart_fallback_fn=fake_chart_fallback,
    )
    assert res["chart_base64"] == "PNG"
    assert captured["data"] == rows  # flat 리스트 그대로 LLM에 전달(래퍼 dict 아님)


def test_roe_single_field_also_uses_chart_fallback():
    """같은 근본원인 확인 — PBR 외 다른 단일 지표(ROE 내림차순)도 동일하게 LLM 폴백으로."""
    rows = [{"stock_code": "005930", "name": "삼성전자", "roe": 12.3}]

    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {"result": rows}}

    called = {}

    def fake_chart_fallback(question, data, llm_fn):
        called["yes"] = True
        return {"chart_base64": "PNG", "chart_title": "T"}

    res = answer_with_verification(
        "ROE 내림차순으로 나열해서 그래프 그려줘", conn=None, llm_fn="fake-llm",
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
        chart_fallback_fn=fake_chart_fallback,
    )
    assert called.get("yes") is True
    assert res["chart_base64"] == "PNG"


# ── chart_llm_fn — 차트 판단을 저가 role="chart" 모델로 (SQL/judge 티어와 분리) ──

def test_uses_chart_llm_fn_when_provided():
    """chart_llm_fn을 주면 chart_fallback_fn은 llm_fn이 아니라 chart_llm_fn을 받는다."""
    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {"result": [{"name": "A", "pbr": 1.0}]}}

    captured = {}

    def fake_chart_fallback(question, data, llm_fn):
        captured["llm_fn"] = llm_fn
        return {"chart_base64": "PNG", "chart_title": "T"}

    res = answer_with_verification(
        "PBR 오름차순 그래프 그려줘", conn=None, llm_fn="SQL-LLM",
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
        chart_fallback_fn=fake_chart_fallback, chart_llm_fn="CHART-LLM",
    )
    assert res["chart_base64"] == "PNG"
    assert captured["llm_fn"] == "CHART-LLM"


def test_chart_llm_fn_defaults_to_llm_fn():
    """chart_llm_fn 미주입(기존 호출부) 시 하위호환 — chart_fallback_fn은 llm_fn을 받는다."""
    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {"result": [{"name": "A", "pbr": 1.0}]}}

    captured = {}

    def fake_chart_fallback(question, data, llm_fn):
        captured["llm_fn"] = llm_fn
        return {"chart_base64": "PNG", "chart_title": "T"}

    answer_with_verification(
        "PBR 오름차순 그래프 그려줘", conn=None, llm_fn="SQL-LLM",
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
        chart_fallback_fn=fake_chart_fallback,
    )
    assert captured["llm_fn"] == "SQL-LLM"


# ── 차트 키워드 게이트 / 불확실 / 폴백 실패 처리 ──────────────────────────────

def test_no_chart_without_keyword():
    """차트 키워드가 없으면 데이터가 있어도 chart_fallback_fn을 아예 호출하지 않는다(오탐 방지)."""
    def stub_route(question, llm_fn):
        return ["backtest"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"backtest": {"result": {"dates": ["a", "b"], "navs": [1.0, 1.1]}}}

    def fake_chart_fallback(question, data, llm_fn):
        raise AssertionError("차트 키워드가 없으면 폴백을 호출하면 안 된다")

    res = answer_with_verification(
        "이 전략 수익률 알려줘", conn=None, llm_fn="fake-llm",
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
        chart_fallback_fn=fake_chart_fallback,
    )
    assert res.get("chart_base64") is None


def test_uses_original_question_for_wants_chart():
    """재시도 피드백이 붙은 dispatch_question이 아니라 원본 question으로 차트 여부를 판단한다."""
    verdicts = iter([{"valid": False, "reason": "1차 실패"}, {"valid": True, "reason": "통과"}])

    def stub_route(question, llm_fn):
        return ["backtest"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"backtest": {"result": {"dates": ["a", "b"], "navs": [1.0, 1.1]}}}

    def flaky_verify(question, domain_results, llm_fn):
        return next(verdicts)

    def fake_chart_fallback(question, data, llm_fn):
        return {"chart_base64": "PNG", "chart_title": "T"}

    res = answer_with_verification(
        "전략 그래프 그려줘", conn=None, llm_fn="fake-llm",
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=flaky_verify,
        chart_fallback_fn=fake_chart_fallback,
    )
    assert res["attempts"] == 2
    assert res.get("chart_base64") == "PNG"  # 원본 질문의 '그려줘'로 차트 생성


def test_uncertain_has_no_chart():
    """3회 실패(불확실) 응답에는 차트 필드가 없다(chart_fallback_fn도 호출 안 함)."""
    def stub_route(question, llm_fn):
        return ["backtest"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"backtest": {"result": {"dates": ["a", "b"], "navs": [1.0, 1.1]}}}

    def always_invalid(question, domain_results, llm_fn):
        return {"valid": False, "reason": "실패"}

    def fake_chart_fallback(question, data, llm_fn):
        raise AssertionError("불확실 응답에는 차트를 만들지 않는다")

    # llm_fn=None: 3회 검증 실패 후의 자유코드 폴백(run_free_exec_fallback)도 llm_fn 없이 즉시
    # 포기 → uncertain=True. 차트 블록은 valid 분기에만 있어 fake_chart_fallback(가드)은 호출되지 않는다.
    res = answer_with_verification(
        "전략 그래프 그려줘", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=always_invalid,
        chart_fallback_fn=fake_chart_fallback,
    )
    assert res["uncertain"] is True
    assert res.get("chart_base64") is None


def test_chart_fallback_returning_none_leaves_no_chart():
    """폴백이 실패(None)하면 차트 없이 텍스트 응답만(에러 아님)."""
    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {"result": [{"stock_code": "005930", "name": "삼성전자", "return_12m": 12.3}]}}

    def fake_chart_fallback(question, data, llm_fn):
        return None

    res = answer_with_verification(
        "코스피 상위 종목 수익률 그래프로 그려줘", conn=None, llm_fn="fake-llm",
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
        chart_fallback_fn=fake_chart_fallback,
    )
    assert res["uncertain"] is False
    assert res.get("chart_base64") is None


def test_default_chart_fallback_with_llm_fn_none_no_chart():
    """기본 chart_fallback_fn(build_chart_freeform)은 llm_fn=None이면 None을 반환 → 차트 없음
    (실제 폴백 경로 end-to-end — 모의 없이 확인, 본문 응답은 정상)."""
    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {"result": [{"name": "A"}]}}

    res = answer_with_verification(
        "저PER 종목 그래프로 보여줘", conn=None, llm_fn=None,
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
    )
    assert res["uncertain"] is False
    assert res.get("chart_base64") is None


# ── _chartable_payload — 폴백에 넘기는 데이터 언랩(도메인 래퍼 제거) ──────────

def test_chart_fallback_receives_unwrapped_flat_list_for_single_domain():
    """단일 도메인 스크리닝: 폴백은 domain_results["kr"]["result"](flat 리스트)를 그대로 받는다.

    래퍼({"kr": {...}})를 넘기면 build_chart_freeform의 요약이 'dict, 최상위 키: [kr]'로만 나와
    LLM이 정작 그릴 리스트를 못 찾아 실행에 실패했다(실측). 멀티턴 경로가 flat한 result를
    넘기는 것과 대칭으로, 여기서도 각 도메인의 result만 꺼내 넘긴다."""
    stocks = [
        {"stock_code": "005930", "name": "삼성전자", "return_12m": 12.3},
        {"stock_code": "000660", "name": "SK하이닉스", "return_12m": 45.6},
    ]

    def stub_route(question, llm_fn):
        return ["kr"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {"intent": "screening", "result": stocks}}

    captured = {}

    def fake_chart_fallback(question, data, llm_fn):
        captured["data"] = data
        return {"chart_base64": "PNG", "chart_title": "T"}

    res = answer_with_verification(
        "코스피 상위 10개 수익률 그래프로 그려줘", conn=None, llm_fn="fake-llm",
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
        chart_fallback_fn=fake_chart_fallback,
    )
    assert res["chart_base64"] == "PNG"
    assert captured["data"] == stocks
    assert isinstance(captured["data"], list)


def test_chart_fallback_receives_per_domain_payloads_for_multi_domain():
    """복합 도메인(kr+macro): 각 도메인의 result만 뽑아 {도메인: payload} dict로 넘긴다
    (도메인별 껍데기 dict가 아니라 그릴 데이터만)."""
    kr_rows = [{"name": "삼성전자", "return_12m": 12.3}]
    macro_rows = [{"as_of": "2026-07-01", "overall": "GREEN"}]

    def stub_route(question, llm_fn):
        return ["kr", "macro"]

    def stub_dispatch(routes, question, conn, llm_fn, steps=None):
        return {"kr": {"result": kr_rows}, "macro": {"result": macro_rows}}

    captured = {}

    def fake_chart_fallback(question, data, llm_fn):
        captured["data"] = data
        return {"chart_base64": "PNG", "chart_title": "T"}

    res = answer_with_verification(
        "한국 상위 종목 수익률이랑 매크로 신호 그래프로 그려줘", conn=None, llm_fn="fake-llm",
        route_fn=stub_route, dispatch_fn=stub_dispatch, verify_fn=_valid_verify,
        chart_fallback_fn=fake_chart_fallback,
    )
    assert res["chart_base64"] == "PNG"
    assert captured["data"] == {"kr": kr_rows, "macro": macro_rows}
