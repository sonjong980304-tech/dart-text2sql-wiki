"""총괄 오케스트레이션 다중 노드 그래프 (Phase 2) — answer_with_verification의 내부 구현.

기존에는 supervisor.answer_with_verification이 순수 파이썬 for 루프로
"라우팅→도메인 실행→검증→(실패 도메인만) 재시도→종합결론"을 처리했다. 이 모듈은 그
제어 흐름을 다이어그램 그대로의 **진짜 LangGraph StateGraph**로 옮긴다:

    START → router → dispatch_gate → (kr | macro | backtest 팬아웃, 병렬) → verify
              │           ↑                                                  │
              │(라우팅 실패)└── 재시도(실패 도메인만)/backtest 추가시도 순환 ──┤
              ↓                                                              │
             END ←──────── fallback(자유 코드 폴백, 정형 전부 실패 시) ←──────┤
                              │(성공 시 재검증 1회는 verify로 순환)           ↓
                              END ←──── chart(조건부) ←──── synthesize(종합결론)

핵심 설계 원칙 — **외부 계약 보존**:
- 이 모듈은 판정/조회 로직을 전혀 새로 만들지 않는다. 라우팅/도메인 실행/검증/종합/
  폴백/차트 함수는 전부 build_supervisor_graph 파라미터로 주입받아 노드 안에서 그대로
  호출한다(supervisor.answer_with_verification이 기본 구현 또는 테스트 주입분을 넘긴다).
- 재시도 상한(max_retries), 실패 도메인만 부분 재-dispatch, 실패 사유 피드백 주입,
  backtest 추가시도 정확히 1회, free_exec 폴백 정확히 1회+재검증 1회, LLM 장애=검증
  불가(통과) 처리, on_progress 이벤트 순서 등 기존 answer_with_verification의 관찰
  가능한 동작을 전부 그대로 재현한다(tests/test_agents_supervisor.py에 박제된 계약).
- 무한루프 없음: 그래프에 사이클(verify→dispatch_gate)이 있지만 verify 노드가
  attempts/phase 상태로 진행을 강제해 정확히 max_retries회(+추가시도 1회)에서 반드시
  탈출한다. 이중 안전망으로 recursion_limit도 호출부(answer_with_verification)가
  max_retries에 비례하는 유한값으로 지정한다.

dispatch 모드 2가지(기존 테스트 주입 계약 때문):
- 팬아웃 모드(use_domain_fanout=True — dispatch_fn 미주입 기본 경로): 도메인마다 자기
  그래프 노드(kr/macro/backtest)가 실행된다. 같은 superstep의 노드들은 LangGraph
  PregelRunner가 스레드로 동시에 실행하므로(pregel/runner.py: "executing a set of
  Pregel tasks concurrently") 도메인 병렬성이 유지된다. 각 도메인 노드는
  dispatch_fn([자기 도메인], ...) 즉 dispatch_domains를 호출한다 — 도메인별 전용
  읽기전용 연결, 예외 흡수, 시작/완료 진행 통지를 전부 그대로 재사용한다.
- 배치 모드(use_domain_fanout=False — 테스트가 dispatch_fn을 주입한 경우): 기존 계약은
  "시도마다 dispatch_fn이 그 시도의 전체 라우트 리스트로 정확히 1회 호출"이다
  (test_answer_with_verification_escalates_...가 dispatch_calls ==
  [["kr"],["kr"],["backtest"]]로 박제). dispatch_gate 노드가 그 1회 호출을 수행하고
  도메인 노드는 건너뛴다 — 주입 스텁이 관찰하는 호출 패턴이 한 글자도 안 바뀐다.

상태 병합 주의: 팬아웃 병렬 쓰기는 reducer(_merge_domain_results)로 병합되는데,
LangGraph는 같은 superstep의 쓰기를 태스크 경로(노드명) 정렬 순으로 적용하므로
(pregel/algo.py apply_writes: "sort tasks on path, to ensure deterministic order")
병합 순서가 노드명 알파벳순(backtest<kr<macro)이 된다. 기존 dispatch_domains는 라우팅
순서(kr→macro→backtest 정규 순서)로 키를 만들었으므로, 소비 시점(ordered_results)에
정규 순서로 재정렬해 하위 소비자(verify/synthesize 프롬프트,
conversation._extract_tabular_data 등)가 보는 키 순서를 기존과 동일하게 유지한다.

Python 3.9 호환(src/agents/graph.py와 동일 사유): LangGraph가 get_type_hints로 런타임
평가하므로 `X | None` 대신 Optional[...]을 쓴다.
"""
from __future__ import annotations

from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

# 정규 도메인 순서이자 도메인 노드 이름(supervisor._DOMAINS와 동일 — 순환 import를 피해
# 상수를 여기 로컬로 둔다. graph.py의 _DOMAIN_LABELS와 같은 관례).
_DOMAIN_NODES: tuple = ("kr", "macro", "backtest")


def _merge_domain_results(left: Optional[Dict[str, Any]], right: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """domain_results 채널 reducer — 팬아웃 병렬 쓰기 병합 + 재시도 시 dict.update 의미론.

    기존 루프의 `domain_results.update(new_results)`와 동일하게, 이미 검증 통과한
    도메인 결과는 보존하고 이번 시도에 (재)실행한 도메인만 덮어쓴다."""
    return {**(left or {}), **(right or {})}


class SupervisorGraphState(TypedDict, total=False):
    # 입력/진행 상태
    question: str                       # 원본 질문(검증/종합결론은 항상 이것을 쓴다)
    dispatch_question: str              # 이번 시도용 질문(재시도 시 실패 피드백이 덧붙음)
    routes: List[str]                   # 최초 라우팅 결과(이후 불변)
    routes_to_dispatch: List[str]       # 이번 시도에 실행할 도메인(1차=전체, 재시도=실패분만)
    attempts: int                       # 정형 재시도 루프 시도 횟수(추가시도/폴백은 안 센다)
    phase: str                          # "formal" | "escalation" | "fallback" — verify 판정 문맥
    last_reason: Optional[str]          # 직전 검증 실패 사유(피드백 주입/폴백 전달용)
    next_action: str                    # verify/fallback 노드가 계산한 다음 행선지(조건부 엣지가 읽음)
    domain_results: Annotated[Dict[str, Any], _merge_domain_results]

    # 최종 산출(answer_with_verification이 반환 dict로 변환)
    uncertain: Optional[bool]
    conclusion: Optional[str]
    reason: Optional[str]
    final_routes: Optional[List[str]]   # 성공 시 반환할 routes(추가시도 성공이면 +["backtest"])
    used_fallback: Optional[bool]
    used_backtest_escalation: Optional[bool]
    chart_base64: Optional[str]
    chart_title: Optional[str]
    charts: Optional[List[Dict[str, Any]]]


def ordered_results(domain_results: Dict[str, Any], canonical: bool) -> Dict[str, Any]:
    """팬아웃 병합(노드명 알파벳순)된 domain_results를 정규 도메인 순서로 재정렬한다.

    배치 모드(canonical=False)에서는 주입된 dispatch_fn이 만든 키 순서를 기존 루프와
    동일하게 그대로 보존해야 하므로 아무것도 하지 않는다. free_exec 등 도메인 외 키는
    기존과 동일하게 뒤에 붙는다."""
    if not canonical:
        return domain_results
    out = {d: domain_results[d] for d in _DOMAIN_NODES if d in domain_results}
    out.update({k: v for k, v in domain_results.items() if k not in out})
    return out


def build_supervisor_graph(
    *,
    conn: Any,
    llm_fn: Optional[Callable[[str], str]],
    max_retries: int,
    route_fn: Callable,
    dispatch_fn: Callable,
    verify_fn: Callable,
    synthesize_fn: Callable,
    fallback_fn: Callable,
    chart_fallback_fn: Callable,
    chart_llm_fn: Optional[Callable[[str], str]],
    wants_chart_fn: Callable[[str], bool],
    chartable_payload_fn: Callable,
    use_domain_fanout: bool,
    steps: Optional[List[Dict[str, Any]]] = None,
    on_progress: Optional[Callable[[str, str], None]] = None,
):
    """총괄 다중 노드 그래프를 조립·컴파일한다(모든 로직 함수는 클로저로 주입).

    conn/llm_fn/steps/on_progress는 상태가 아니라 클로저로 바인딩한다(직렬화 불가한
    객체를 상태 채널에 태우지 않는 기존 graph.py build_hierarchical_graph 관례와 동일).
    """
    # on_progress가 없으면 하위 함수에 그 키워드 자체를 안 넘긴다 — 주입되는 fake
    # route_fn/dispatch_fn(테스트)이 on_progress 파라미터를 몰라도 깨지지 않는다(기존 동일).
    progress_kwargs = {"on_progress": on_progress} if on_progress else {}

    def _results(state: SupervisorGraphState) -> Dict[str, Any]:
        return ordered_results(state.get("domain_results") or {}, use_domain_fanout)

    # ── router: 라우팅은 정확히 1회 ─────────────────────────────────────────────
    def router_node(state: SupervisorGraphState) -> dict:
        question = state["question"]
        routes = route_fn(question, llm_fn, **progress_kwargs)
        if not routes:
            # 라우팅이 도메인을 하나도 못 찾음(unknown) — dispatch/verify 낭비 없이 즉시
            # 불확실 응답(기존 answer_with_verification의 빈 라우트 조기 반환과 동일).
            if on_progress:
                on_progress("supervisor", "질문을 이해하지 못했습니다 — 처리 가능한 도메인을 찾지 못함")
            return {
                "routes": [],
                "attempts": 0,
                "uncertain": True,
                "reason": "질문을 이해하지 못했습니다. 국내 주식, 매크로 지표, 백테스트 중 "
                          "어떤 것에 대한 질문인지 좀 더 구체적으로 말씀해 주세요.",
                "next_action": "end",
            }
        return {
            "routes": routes,
            "routes_to_dispatch": list(routes),
            "attempts": 0,
            "phase": "formal",
            "next_action": "dispatch",
        }

    def route_after_router(state: SupervisorGraphState) -> str:
        return "end" if state["next_action"] == "end" else "dispatch"

    # ── dispatch_gate: 시도 준비(피드백 주입·시도 횟수) + 배치 모드 실행 ──────────
    def dispatch_gate_node(state: SupervisorGraphState) -> dict:
        question = state["question"]
        updates: dict = {}
        if state["phase"] == "formal":
            updates["attempts"] = state.get("attempts", 0) + 1
            # 재시도(2회차부터)에는 직전 실패 사유를 도메인 실행 질문에 피드백으로 덧붙인다.
            # 검증/종합결론은 피드백이 섞이지 않은 원본 question을 쓴다(기존 동일).
            dispatch_question = question
            last_reason = state.get("last_reason")
            if last_reason:
                dispatch_question = (
                    f"{question}\n\n"
                    f"[이전 시도 실패 피드백] 직전 시도가 다음 이유로 검증에 실패했습니다: {last_reason}\n"
                    f"같은 방식을 그대로 반복하지 말고, 이 피드백을 반영해 다른 접근으로 다시 답하세요."
                )
        else:
            # backtest 추가시도(escalation)는 피드백 없는 원본 질문으로 1회만(기존 동일),
            # attempts도 정형 루프 것을 그대로 유지한다.
            dispatch_question = question
        updates["dispatch_question"] = dispatch_question
        if not use_domain_fanout:
            # 배치 모드: 주입된 dispatch_fn을 "이번 시도의 전체 라우트로 정확히 1회" 호출
            # (기존 루프의 호출 패턴 그대로 — 주입 스텁의 관찰 동작 보존).
            new_results = dispatch_fn(
                state["routes_to_dispatch"], dispatch_question, conn, llm_fn,
                steps=steps, **progress_kwargs,
            )
            updates["domain_results"] = new_results
        return updates

    def route_after_gate(state: SupervisorGraphState):
        if not use_domain_fanout:
            return "verify"
        targets = [d for d in state.get("routes_to_dispatch") or [] if d in _DOMAIN_NODES]
        # 알 수 없는 도메인뿐이면(기존 dispatch_domains가 빈 결과를 내던 경우와 동일하게)
        # 도메인 실행 없이 곧장 검증으로 — verify가 결정론적으로 invalid 처리한다.
        return targets or "verify"

    # ── 도메인 노드(kr/macro/backtest): 팬아웃 병렬 실행 ─────────────────────────
    def make_domain_node(domain: str):
        def domain_node(state: SupervisorGraphState) -> dict:
            # dispatch_domains([domain], ...) 재사용 — 도메인별 전용 읽기전용 연결,
            # 예외 흡수({"error": ...}), 시작/완료 on_progress 통지가 전부 기존 그대로다.
            new_results = dispatch_fn(
                [domain], state["dispatch_question"], conn, llm_fn,
                steps=steps, **progress_kwargs,
            )
            return {"domain_results": new_results}

        return domain_node

    # ── verify: 검증 + 다음 행선지 결정(재시도 순환/추가시도/폴백/종합결론) ────────
    def verify_node(state: SupervisorGraphState) -> dict:
        question = state["question"]
        routes = state["routes"]
        attempts = state.get("attempts", 0)
        phase = state["phase"]
        verdict = verify_fn(question, _results(state), llm_fn)

        if phase == "formal":
            if verdict.get("valid"):
                if on_progress:
                    on_progress("verify", f"{attempts}차 검증 통과")
                return {"next_action": "synthesize", "final_routes": list(routes)}
            last_reason = verdict.get("reason")
            if attempts < max_retries:
                # 실패한 도메인만 부분 재-dispatch(성공 도메인 결과는 reducer가 보존).
                # per_domain 정보가 없는 verify_fn(단순 fake)은 전체 재-dispatch — 하위호환.
                per_domain = verdict.get("per_domain") or {}
                if per_domain:
                    routes_to_dispatch = [
                        d for d in routes if not per_domain.get(d, {}).get("valid", False)
                    ]
                else:
                    routes_to_dispatch = list(routes)
                if on_progress:
                    on_progress("verify", f"{attempts}차 검증 실패: {last_reason} → 재시도")
                return {
                    "next_action": "retry",
                    "last_reason": last_reason,
                    "routes_to_dispatch": routes_to_dispatch,
                }
            if on_progress:
                on_progress("verify", f"{attempts}차 검증 실패: {last_reason}")
            if "backtest" not in routes:
                # 정형 루프 소진 + 원래 라우팅에 backtest 없음 → 추가시도 정확히 1회(기존 동일).
                if on_progress:
                    on_progress("supervisor", f"{max_retries}회 정형 검증 실패 → backtest 도메인 추가시도")
                return {
                    "next_action": "escalate",
                    "last_reason": last_reason,
                    "phase": "escalation",
                    "routes_to_dispatch": ["backtest"],
                }
            return {"next_action": "fallback", "last_reason": last_reason}

        if phase == "escalation":
            if verdict.get("valid"):
                if on_progress:
                    on_progress("verify", "backtest 추가시도 검증 통과")
                return {
                    "next_action": "synthesize",
                    "used_backtest_escalation": True,
                    "final_routes": list(routes) + ["backtest"],
                }
            last_reason = verdict.get("reason")
            if on_progress:
                on_progress("verify", f"backtest 추가시도도 검증 실패: {last_reason}")
            return {"next_action": "fallback", "last_reason": last_reason}

        # phase == "fallback": free_exec 결과 재검증 정확히 1회. 실패해도 결과를 버리지
        # 않고 verification_warning만 남긴다(최후 수단이라 대안이 없음 — 기존 동일).
        if verdict.get("valid"):
            if on_progress:
                on_progress("verify", "자유 코드 폴백 결과 재검증 통과")
            return {"next_action": "synthesize_fallback"}
        reason = verdict.get("reason")
        if on_progress:
            on_progress("verify", f"자유 코드 폴백 결과 재검증 실패(참고용으로 표시): {reason}")
        free_exec = dict((state.get("domain_results") or {}).get("free_exec") or {})
        free_exec["verification_warning"] = reason
        return {"next_action": "synthesize_fallback", "domain_results": {"free_exec": free_exec}}

    def route_after_verify(state: SupervisorGraphState) -> str:
        return state["next_action"]

    # ── fallback: 자유 코드 생성 폴백(정확히 1회, 재시도 없음) ────────────────────
    def fallback_node(state: SupervisorGraphState) -> dict:
        last_reason = state.get("last_reason")
        fallback = fallback_fn(state["question"], conn, llm_fn, last_reason)
        if fallback.get("ok"):
            if on_progress:
                on_progress("supervisor", f"{max_retries}회 정형 검증 실패 → 자유 코드 생성 폴백 성공")
            return {
                "next_action": "verify",
                "phase": "fallback",
                "used_fallback": True,
                "domain_results": {
                    "free_exec": {
                        "fallback_used": True,
                        "sql": fallback.get("sql"),
                        "code": fallback.get("code"),
                        "result": fallback.get("result"),
                    }
                },
            }
        if on_progress:
            on_progress(
                "supervisor",
                f"{max_retries}회 정형 검증 실패 → 자유 코드 생성 폴백도 실패: {fallback.get('error')}",
            )
        return {
            "next_action": "end",
            "uncertain": True,
            "reason": (
                f"{max_retries}회 검증에 모두 실패했습니다. 확실한 답을 제시할 수 없습니다."
                + (f" (마지막 사유: {last_reason})" if last_reason else "")
                + (f" (자유 코드 생성 폴백도 실패: {fallback.get('error')})" if fallback.get("error") else "")
            ),
        }

    def route_after_fallback(state: SupervisorGraphState) -> str:
        return "verify" if state["next_action"] == "verify" else "end"

    # ── synthesize: 종합결론(검증 통과/폴백 성공 시에만 — 실패 경로는 진입 안 함) ──
    def synthesize_node(state: SupervisorGraphState) -> dict:
        conclusion = synthesize_fn(state["question"], _results(state), llm_fn)
        return {"conclusion": conclusion, "uncertain": False}

    def route_after_synthesize(state: SupervisorGraphState) -> str:
        # 폴백 성공 경로는 기존과 동일하게 차트를 붙이지 않는다. 차트 요청 판단은 재시도
        # 피드백이 섞이지 않은 원본 question 기준(기존 동일).
        if state["next_action"] != "synthesize_fallback" and wants_chart_fn(state["question"]):
            return "chart"
        return "end"

    # ── chart: 명시적 차트 요청 시에만(조건부 진입) LLM 자유 차트 폴백 ────────────
    def chart_node(state: SupervisorGraphState) -> dict:
        chart = chart_fallback_fn(
            state["question"], chartable_payload_fn(_results(state)), chart_llm_fn or llm_fn
        )
        if not chart:
            # 부가 기능 — 실패해도 본문 응답을 무너뜨리지 않는다(기존 동일).
            return {}
        b64, title = chart["chart_base64"], chart.get("chart_title")
        return {
            "chart_base64": b64,
            "chart_title": title,
            "charts": [{"chart_base64": b64, "chart_title": title}],
        }

    g = StateGraph(SupervisorGraphState)
    g.add_node("router", router_node)
    g.add_node("dispatch_gate", dispatch_gate_node)
    for domain in _DOMAIN_NODES:
        g.add_node(domain, make_domain_node(domain))
        g.add_edge(domain, "verify")  # 팬아웃된 도메인들이 전부 끝난 뒤 verify 1회 실행(BSP)
    g.add_node("verify", verify_node)
    g.add_node("fallback", fallback_node)
    g.add_node("synthesize", synthesize_node)
    g.add_node("chart", chart_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", route_after_router, {"dispatch": "dispatch_gate", "end": END})
    # 팬아웃 모드: 이번 시도의 도메인 노드 이름 리스트를 반환 → 같은 superstep에서 병렬 실행.
    # 배치 모드: 도메인 노드를 건너뛰고 곧장 verify.
    g.add_conditional_edges(
        "dispatch_gate", route_after_gate,
        {"verify": "verify", **{d: d for d in _DOMAIN_NODES}},
    )
    g.add_conditional_edges(
        "verify", route_after_verify,
        {
            "retry": "dispatch_gate",       # 순환: 실패한 도메인만 재-dispatch
            "escalate": "dispatch_gate",    # 순환: backtest 추가시도 1회
            "fallback": "fallback",
            "synthesize": "synthesize",
            "synthesize_fallback": "synthesize",
        },
    )
    g.add_conditional_edges("fallback", route_after_fallback, {"verify": "verify", "end": END})
    g.add_conditional_edges("synthesize", route_after_synthesize, {"chart": "chart", "end": END})
    g.add_edge("chart", END)
    return g.compile()
