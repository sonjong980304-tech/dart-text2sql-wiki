"""1회성 실측 스크립트 — 같은 goldset 문항을 여러 LLM 모델로 돌려 정확도/속도를 비교한다.

사용:
  python3 scripts/eval_model_comparison.py --limit 8
  python3 scripts/eval_model_comparison.py --limit 8 --models gpt-5.4-mini,exaone3.5:7.8b
  python3 scripts/eval_model_comparison.py --limit 8 --judge-model gpt-5.5

동작:
  - 후보 모델 각각에 대해 LLMClient(model=후보)를 llm_fn 으로 고정해 같은 goldset subset을
    run_hierarchical 로 재실행한다(scripts/eval_hierarchical_goldset.py의 run_goldset과
    동일한 판정 로직 재사용 — src/eval/hierarchical_runner.run_hierarchical_eval).
    model override가 있으면 LLMClient.model_for()가 role(sql/judge/...)과 무관하게 그
    모델을 그대로 쓰므로(src/llm.py), 라우팅·SQL생성·검증·종합결론 전 단계가 후보 모델
    하나로 일관되게 실행된다.
  - judge(채점) 모델은 비교 대상과 분리해 항상 고정한다(기본 gpt-5.5) — 채점 기준이
    모델마다 흔들리면 정확도 비교 자체가 무의미해지므로.
  - 원본 DB는 항상 격리 사본에서만 읽는다(run_hierarchical_eval이 내부에서 처리).

주의: OpenAI 모델은 실제 API를 호출한다(비용 발생). 로컬(ollama) 모델은 Mac 로컬에서
추론하므로 문항당 응답시간이 GPT보다 훨씬 길 수 있다(실측 확인 필요).
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agents.graph import run_hierarchical
from src.config import CONFIG
from src.db import connect_readonly
from src.eval.goldset import GOLDSET
from src.eval.hierarchical_runner import run_hierarchical_eval
from src.llm import LLMClient

RESEARCH_DIR = Path(__file__).resolve().parent.parent / ".omc" / "research"

_DEFAULT_MODELS = [
    "gpt-5.4-mini",
    "gpt-5.5",
    "exaone3.5:7.8b",
    "qwen2.5-coder:7b-instruct-q4_K_M",
]


def _llm_fn_for(model: str):
    client = LLMClient(model=model)
    if not client.available:
        return None
    return lambda prompt: (client.complete(prompt, role="sql").text or "")


def _fmt_report(results: dict[str, dict], n: int, judge_model: str) -> str:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        "# 모델별 goldset 정확도/속도 비교",
        "",
        f"- 실측 일시: {ts}",
        f"- 실행 문항 수: {n} (goldset 전체 {len(GOLDSET)}문항 중 앞 {n}문항)",
        f"- 채점(judge) 모델: {judge_model} (비교 대상과 분리해 고정)",
        f"- DB: 원본({CONFIG.db_path})의 격리 사본에서 실행(원본 보호)",
        "",
        "## 요약",
        "",
        "| 모델 | 가용 | 정확도(judge match) | 평균 응답시간 | 평균 LLM호출 |",
        "|---|---|---|---|---|",
    ]
    for model, r in results.items():
        if r is None:
            lines.append(f"| {model} | 미가용(스킵) | - | - | - |")
            continue
        acc = f"{r['accuracy_pct']}% ({r['match']}/{r['judged']})" if r["judged"] else "판정불가"
        lines.append(
            f"| {model} | O | {acc} | {r['avg_latency_s']}s | {r['avg_llm_calls']}회 |"
        )

    lines += ["", "## 문항별 상세", ""]
    for model, r in results.items():
        if r is None:
            continue
        lines.append(f"### {model}")
        lines.append("")
        lines.append("| id | tags | 질문 | 라우팅 | 불확실 | match | 응답(s) | LLM호출 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for row in r["rows"]:
            q = (row["question"] or "").replace("|", "/")[:40]
            match = "✅" if row["match"] else ("❌" if row["match"] is False else "-")
            lines.append(
                f"| {row['id']} | {row['tags']} | {q} | {row['routes']} | "
                f"{'Y' if row['uncertain'] else ''} | {match} | {row['latency_s']} | {row['llm_calls']} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8, help="goldset 재실행 문항 수(앞에서부터)")
    ap.add_argument("--models", type=str, default=",".join(_DEFAULT_MODELS), help="비교할 모델 id(콤마 구분)")
    ap.add_argument("--judge-model", type=str, default="gpt-5.5", help="채점 전용 고정 모델")
    args = ap.parse_args()

    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    items = GOLDSET[: args.limit]

    judge_client = LLMClient(model=args.judge_model)
    if not judge_client.available:
        print(f"[경고] judge 모델({args.judge_model}) 미가용 — 판정이 전부 판정불가로 나올 수 있음")
    judge_llm_fn = (lambda prompt: (judge_client.complete(prompt, role="judge").text or "")) if judge_client.available else None

    results: dict[str, dict | None] = {}
    for model in models:
        llm_fn = _llm_fn_for(model)
        if llm_fn is None:
            print(f"[스킵] {model}: 미가용(API 키 없음 또는 ollama 데몬에 모델 없음)")
            results[model] = None
            continue
        print(f"[{model}] {len(items)}문항 재실행 시작 …")
        r = run_hierarchical_eval(
            items, run_hierarchical, llm_fn=llm_fn, judge_llm_fn=judge_llm_fn,
            db_path=CONFIG.db_path, connect_ro_fn=connect_readonly,
        )
        results[model] = r
        print(f"[{model}] 완료: 정확도={r['accuracy_pct']}% 평균응답={r['avg_latency_s']}s 평균호출={r['avg_llm_calls']}회")

    md = _fmt_report(results, len(items), args.judge_model)
    out = RESEARCH_DIR / "model-comparison-goldset.md"
    out.write_text(md, encoding="utf-8")
    print(f"[완료] 기록: {out}")


if __name__ == "__main__":
    main()
