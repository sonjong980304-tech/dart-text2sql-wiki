"""계층형 총괄 그래프 (HA-11) — HA-10의 총괄 로직을 LangGraph StateGraph 노드로 감싼다.

.omc/state/sessions/f6b79d90-b7b2-4b4f-b4f7-405a8355724e/prd.json 의 HA-11 참고.

HA-10(src/agents/supervisor.py)은 라우팅·도메인실행·정합성검증·재시도를 이미
`answer_with_verification(...)` 하나로 통합해 두었다. 이 파일은 **새 로직을 만들지 않고**
그 총괄 함수를 LangGraph 노드(`supervisor_node`)로 감싸 StateGraph 로 조립하고, 실행을
`.invoke()`가 아니라 `.stream()`으로 돌려 노드 완료 시점마다 진행 이벤트를 방출한다.

--- AC5(코드리뷰) 근거 --------------------------------------------------------
"총괄 에이전트의 라우팅·검증 로직이 LangGraph 노드로 구현되어 있다"는 것이 핵심이다.
그 라우팅(route_question)·검증(verify_answer)·재시도(answer_with_verification) 로직은
전부 supervisor_node 안에서 answer_with_verification 호출로 실행된다. 총괄 노드가 이미
route→dispatch→verify→retry 를 내부에서 수행하므로, 이 스토리에서는 그래프를 단일 노드
(START→supervisor→END)로 둔다 — 도메인 실행 자체는 dispatch_domains 내부 순수 함수
호출로 처리되므로 그래프 레벨에서 도메인마다 노드를 쪼갤 필요가 없다. 중요한 것은
`.stream()`으로 노드 완료마다 이벤트가 나오는 것이다.

--- 스트리밍 이벤트 스키마(HA-12 SSE 인계용) ----------------------------------
각 노드 완료 이벤트는 **단계 이름 + 핵심 결과 한 줄**만 담는다(SQL 전문/원본 rows/
결론 본문 같은 상세는 넣지 않는다).
    {"step": "supervisor", "summary": "한국+백테스트 도메인 라우팅, 검증 통과(2회 시도)"}
최종 결론/원본 도메인 결과는 이벤트가 아니라 그래프 최종 상태로 얻는다 — 동기 호출부는
run_hierarchical, 스트리밍 호출부는 run_streaming(..., out_final={})로 같은 실행 한 번에서
받는다(진행상황용/최종답변용으로 그래프를 두 번 돌리지 않기 위함, HA-12 후속 수정).

--- 상태 주의(Python 3.9 호환) ------------------------------------------------
src/graph/state.py 와 동일하게 LangGraph가 get_type_hints 로 런타임 평가하므로
`X | None` 대신 Optional[...] 을 쓴다.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Callable, Dict, Iterator, List, Optional, TypedDict

from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph

from src.agents.supervisor import answer_with_verification

# 도메인 코드 → 이벤트 요약용 짧은 한글 라벨.
_DOMAIN_LABELS: Dict[str, str] = {
    "kr": "한국",
    "macro": "매크로",
    "backtest": "백테스트",
}


class HierarchicalState(TypedDict, total=False):
    # 입력
    question: str                      # 사용자 질문
    steps: List[Dict[str, Any]]        # 백테스트 파이프라인 스텝(있으면 backtest 도메인이 소비)
    conn: Any                          # DB 연결(노드에 클로저로 주입되지 않을 때 state 폴백)
    llm_fn: Any                        # Callable[[str], str] (동일 폴백용)
    chart_llm_fn: Any                  # 차트 판단 전용 llm_fn(없으면 llm_fn 재사용) — web가 저가 role="chart"로 주입
    on_progress: Any                   # Callable[[str, str], None] — 실시간 진행 콜백(HA-12 확장)

    # 총괄 노드 산출
    routes: List[str]                  # 라우팅된 도메인(["kr"] | ["kr","backtest"] ...)
    domain_results: Dict[str, Any]     # 도메인별 원본 결과(가공 없음)
    conclusion: Optional[str]          # 검증 통과 시 종합결론
    uncertain: Optional[bool]          # 검증 실패(재시도 소진, 기본 max_retries=2) 시 True
    attempts: Optional[int]            # 실제 시도 횟수
    reason: Optional[str]              # 불확실 사유(실패 시)
    chart_base64: Optional[str]        # 차트 요청 시 이미지(PNG base64, 접두사 없음), 아니면 None
    chart_title: Optional[str]         # 차트 제목(차트가 있을 때만)
    charts: Optional[List[Dict[str, Any]]]  # 차트가 여러 개(산점도+막대그래프 등)면 전부. 없으면 None
    used_fallback: Optional[bool]      # 정형 검증 재시도 소진 후 자유 코드 생성 폴백으로 답했으면 True

    # 스트리밍 이벤트 누적(노드 완료 시점마다 append). 여러 노드로 확장돼도 누적되도록 reducer 지정.
    events: Annotated[List[Dict[str, Any]], operator.add]


def _summarize(result: dict) -> str:
    """총괄 결과 → 이벤트 요약 한 줄. 라우팅 도메인 + 검증 통과/실패 + 시도 횟수만 담는다.

    (SQL 전문/원본 rows/결론 본문 등 상세는 의도적으로 제외 — HA-12가 SSE로 그대로 흘려도
    민감/장문 데이터가 새지 않게 한다.)
    """
    routes = result.get("routes") or []
    labels = "+".join(_DOMAIN_LABELS.get(r, r) for r in routes)
    attempts = result.get("attempts")
    if result.get("uncertain"):
        status = f"검증 실패({attempts}회 시도, 불확실)"
    else:
        status = f"검증 통과({attempts}회 시도)"
    prefix = f"{labels} 도메인 라우팅" if labels else "도메인 라우팅"
    return f"{prefix}, {status}"


def _make_stream_progress() -> Optional[Callable[..., None]]:
    """LangGraph 표준 스트리밍(get_stream_writer)으로 진행 이벤트를 방출하는 on_progress
    콜백을 만든다.

    총괄/도메인 로직은 여전히 `on_progress(step, summary, detail=None)` 콜백 규약으로 진행
    상황을 알린다(그 내부는 변경하지 않는다). 달라진 것은 그 콜백의 "배선"뿐이다 — 예전엔
    run_streaming 이 스레드+큐로 콜백을 소비했지만, 이제는 이 콜백이 get_stream_writer()의
    StreamWriter 로 커스텀 이벤트({"step","summary"[,"detail"]})를 흘려보내고, .stream(
    stream_mode="custom") 이 그걸 실시간으로 방출한다.

    StreamWriter(내부적으로 SyncQueue.put — deque.append + Semaphore.release)는 스레드
    안전하므로, dispatch_domains 가 도메인들을 병렬 스레드로 실행하며 각자 on_progress 를
    호출해도 안전하다(라이터는 노드 컨텍스트에서 한 번만 해석해 클로저로 넘기므로 자식
    스레드가 get_stream_writer 를 다시 부를 필요가 없다).

    그래프 실행 컨텍스트 밖(예: supervisor_node 직접 호출)에서는 get_stream_writer 가
    RuntimeError 를 던지므로 None 을 돌려준다 — 이 경우 on_progress 없이 기존과 완전히
    동일하게 동작한다(회귀 없음). custom 스트림 모드가 아닌 실행(run_hierarchical 의
    stream_mode="values")에서는 no-op StreamWriter 가 반환돼 이벤트가 조용히 무시된다.
    """
    try:
        writer = get_stream_writer()
    except RuntimeError:
        return None

    def on_progress(step: str, summary: str, detail: Optional[Dict[str, Any]] = None) -> None:
        event: Dict[str, Any] = {"step": step, "summary": summary}
        if detail is not None:
            event["detail"] = detail
        writer(event)

    return on_progress


def supervisor_node(
    state: HierarchicalState,
    conn: Any = None,
    llm_fn: Optional[Callable[[str], str]] = None,
    chart_llm_fn: Optional[Callable[[str], str]] = None,
) -> dict:
    """총괄 에이전트를 감싼 LangGraph 노드.

    HA-10의 answer_with_verification 을 호출한다 — 라우팅(route_question)·도메인 실행
    (dispatch_domains)·정합성 검증(verify_answer)·최대 max_retries(기본 2)회 재시도가 이
    한 번의 호출 안에서 모두 수행된다. 즉 총괄 에이전트의 라우팅·검증 로직이 이 노드로
    구현된 것이다(AC5).

    conn/llm_fn 은 build_hierarchical_graph 가 클로저로 주입한다. 인자로 안 넘어오면
    state 에서 폴백으로 읽어(직접 노드 등록도 지원) answer_with_verification 에 전달한다.

    반환은 state 갱신분(dict) — routes/domain_results/conclusion/uncertain/attempts/reason
    과 이번 노드의 진행 이벤트 한 건을 events 로 돌려준다(events reducer가 누적).
    """
    question = state["question"]
    conn = conn if conn is not None else state.get("conn")
    llm_fn = llm_fn if llm_fn is not None else state.get("llm_fn")
    chart_llm_fn = chart_llm_fn if chart_llm_fn is not None else state.get("chart_llm_fn")
    steps = state.get("steps")
    # 진행 이벤트 배선: state 로 on_progress 를 직접 주입한 호출부가 있으면 그것을 우선
    # 존중하고(하위호환), 없으면 LangGraph 표준 스트리밍(get_stream_writer) 기반 콜백을
    # 만든다. 그래프 밖 직접 호출이면 None 이 되어 기존과 동일하게 통지 없이 동작한다.
    on_progress = state.get("on_progress")
    if on_progress is None:
        on_progress = _make_stream_progress()

    # chart_llm_fn은 값이 있을 때만 넘긴다(answer_with_verification 자체의 on_progress 선택적
    # 전달 관례와 동일) — 이 노드를 감싸는 테스트/직접 호출부의 fake answer_with_verification이
    # 이 선택적 인자를 몰라도 깨지지 않게 한다. web는 항상 저가 role="chart" llm_fn을 주입한다.
    chart_kwargs = {"chart_llm_fn": chart_llm_fn} if chart_llm_fn is not None else {}
    result = answer_with_verification(
        question, conn, llm_fn, steps=steps, on_progress=on_progress, **chart_kwargs,
    )

    event = {"step": "supervisor", "summary": _summarize(result)}
    return {
        "routes": result.get("routes", []),
        "domain_results": result.get("domain_results", {}),
        "conclusion": result.get("conclusion"),
        "uncertain": result.get("uncertain"),
        "attempts": result.get("attempts"),
        "reason": result.get("reason"),
        # 차트 요청 시에만 채워지는 필드(그대로 pass-through — web/app.py가 {**result}로 노출).
        "chart_base64": result.get("chart_base64"),
        "chart_title": result.get("chart_title"),
        "charts": result.get("charts"),
        "used_fallback": result.get("used_fallback"),
        "events": [event],
    }


def build_hierarchical_graph(
    conn: Any,
    llm_fn: Optional[Callable[[str], str]] = None,
    chart_llm_fn: Optional[Callable[[str], str]] = None,
):
    """총괄 노드 하나로 StateGraph 를 조립·컴파일한다: START → supervisor → END.

    supervisor_node 에 conn/llm_fn 을 클로저로 바인딩해 등록한다(기존 src/graph/build.py 의
    make_nodes(deps) 바인딩 관례와 동일 철학). 반환은 컴파일된 그래프(.stream()/.invoke() 지원).
    """
    def _supervisor(state: HierarchicalState) -> dict:
        return supervisor_node(state, conn=conn, llm_fn=llm_fn, chart_llm_fn=chart_llm_fn)

    g = StateGraph(HierarchicalState)
    g.add_node("supervisor", _supervisor)
    g.add_edge(START, "supervisor")
    g.add_edge("supervisor", END)
    return g.compile()


def run_streaming(
    question: str,
    conn: Any,
    llm_fn: Optional[Callable[[str], str]] = None,
    steps: Optional[List[Dict[str, Any]]] = None,
    out_final: Optional[Dict[str, Any]] = None,
    chart_llm_fn: Optional[Callable[[str], str]] = None,
) -> Iterator[dict]:
    """그래프를 `.stream(stream_mode=["custom","values"])`로 실행하며 노드 내부 진행 이벤트를
    실시간으로 하나씩 yield 한다(HA-12 확장 — 실시간 트리 상세화).

    예전 구현은 별도 워커 스레드 + queue.Queue + error_box 로 on_progress 콜백을 손수
    배선했다(당시 `.stream()`을 그냥 쓰면 "노드 완료 후 요약 한 줄"만 나와 실시간이 아니었기
    때문). 지금은 LangGraph 표준 스트리밍으로 대체한다 — supervisor_node 가 get_stream_writer()
    로 만든 on_progress 가 진행마다 커스텀 이벤트({"step","summary"[,"detail"]})를 StreamWriter
    에 흘려보내고, stream_mode 에 "custom" 을 포함해 `.stream()`을 직접 소비하면 그 이벤트가
    생성되는 즉시 방출된다. custom 이벤트는 노드 실행 중 실시간으로, 최종 상태(values)는
    superstep 종료 후에 도착하므로, 커스텀 이벤트만 yield 하고 마지막 values 스냅샷을 최종
    상태로 잡는다. 스레드/큐 배선이 사라져 코드가 단순해지고, 스레드 간 예외 전달(error_box)도
    불필요해졌다(예외는 이 제너레이터를 소비하는 쪽으로 자연스럽게 전파된다).

    이벤트 스키마({"step","summary"[,"detail"]})는 예전과 동일하게 유지되므로 web/app.py 의
    SSE 소비부(GET /api/query/stream)가 받는 이벤트 모양은 바뀌지 않는다.

    out_final: 전달하면(가변 dict, 예: {}) 이 실행이 정상 종료된 뒤 그래프 최종 상태
    (routes/domain_results/conclusion/uncertain/attempts/...)로 채워진다. 이 실행 하나가
    진행 이벤트와 최종 답변을 모두 내어주므로, 호출부가 "진행상황용"과 "최종답변용"을
    별도로 두 번 실행할 필요가 없다(web/app.py의 GET /api/query/stream이 과거 이 최종
    상태를 버리고 POST /api/query를 한 번 더 호출해 동일 질문을 두 번 계산하던 문제의
    해결책 — 진행 이벤트(custom)는 실시간 통지로만 쓰고, 결과는 out_final로 받는다).
    """
    graph = build_hierarchical_graph(conn, llm_fn, chart_llm_fn)
    init: dict = {"question": question}
    if steps is not None:
        init["steps"] = steps
    final_snapshot: dict = dict(init)
    # stream_mode 가 리스트이면 (mode, payload) 튜플로 방출된다(langgraph pregel.output()).
    for mode, chunk in graph.stream(init, stream_mode=["custom", "values"]):
        if mode == "custom":
            yield chunk               # 진행 이벤트({"step","summary"[,"detail"]}) 실시간 방출
        else:                          # "values": 각 superstep 종료 후 누적 상태 스냅샷
            final_snapshot = chunk
    if out_final is not None:
        out_final.update(final_snapshot)


def collect_stream(
    question: str,
    conn: Any,
    llm_fn: Optional[Callable[[str], str]] = None,
    steps: Optional[List[Dict[str, Any]]] = None,
) -> List[dict]:
    """run_streaming 을 리스트로 모아 반환하는 테스트/동기 소비용 버전."""
    return list(run_streaming(question, conn, llm_fn=llm_fn, steps=steps))


def run_hierarchical(
    question: str,
    conn: Any,
    llm_fn: Optional[Callable[[str], str]] = None,
    steps: Optional[List[Dict[str, Any]]] = None,
    chart_llm_fn: Optional[Callable[[str], str]] = None,
) -> dict:
    """그래프를 `.stream()`(values 모드)으로 실행해 **최종 누적 상태**(dict)를 반환한다.

    스트리밍 이벤트(진행 표시)와 별개로 최종 결론/원본 도메인 결과가 필요한 호출부(HA-12의
    최종 SSE 메시지 등)를 위해, invoke 대신 stream 마지막 스냅샷을 최종 상태로 돌려준다.
    """
    graph = build_hierarchical_graph(conn, llm_fn, chart_llm_fn)
    init: dict = {"question": question}
    if steps is not None:
        init["steps"] = steps
    final: dict = dict(init)
    for snapshot in graph.stream(init, stream_mode="values"):
        final = snapshot
    return final
