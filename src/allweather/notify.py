"""올웨더 텔레그램 알림 (델타 포함).

.omc/specs/brainstorming-all-weather-portfolio.md AC12/AC13 참고.

- 전송은 quant_trader notifier.send_telegram과 동일한 방식(raw requests.post, HTML parse_mode)
  이며 동일한 봇/채팅방(env var TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID)을 재사용한다. quant_trader
  코드를 import하지 않고 같은 패턴으로 새로 작성했다(env var는 그대로).
- 메시지에는 이번 달 비중과 함께, 직전 달 저장값 대비 변경분(델타, 예: "삼성전자 25.0%→30.0%(+5.0%p)")을
  포함한다(AC13). 직전 행이 없으면(첫 실행) 현재 비중만 표기한다.
"""
from __future__ import annotations

import logging
import os

import requests

from .data import TICKERS

logger = logging.getLogger(__name__)


def _name(ticker: str) -> str:
    return TICKERS.get(ticker, ticker)


def build_delta_message(current: dict, previous: dict | None) -> str:
    """이번 스냅샷 + (직전 대비) 비중 델타 + 성과지표 요약 메시지(HTML)."""
    cur_w = current.get("weights", {})
    prev_w = (previous or {}).get("weights", {})

    lines = ["🌦️ <b>올웨더 포트폴리오 리밸런싱</b>"]
    computed_at = current.get("computed_at")
    if computed_at:
        lines.append(f"기준일: {computed_at}")
    lines.append("")
    lines.append("<b>목표 비중</b>")

    # 티커 순서는 TICKERS 정의 순서를 따르되, 없는 티커는 뒤에 붙인다.
    ordered = list(TICKERS.keys()) + [t for t in cur_w if t not in TICKERS]
    for t in ordered:
        if t not in cur_w:
            continue
        cw = cur_w[t] * 100
        if t in prev_w:
            pw = prev_w[t] * 100
            delta = cw - pw
            sign = "+" if delta >= 0 else ""
            lines.append(f"  {_name(t)}: {pw:.1f}%→{cw:.1f}%({sign}{delta:.1f}%p)")
        else:
            lines.append(f"  {_name(t)}: {cw:.1f}%")

    lines.append("")
    lines.append(
        "CAGR {cagr:.1f}% · MDD {mdd:.1f}% · 샤프 {sharpe:.2f} · 누적 {cum:.1f}%".format(
            cagr=(current.get("cagr") or 0) * 100,
            mdd=(current.get("mdd") or 0) * 100,
            sharpe=(current.get("sharpe") or 0),
            cum=(current.get("cumulative_return") or 0) * 100,
        )
    )
    return "\n".join(lines)


def send_telegram(message: str) -> bool:
    """텔레그램 봇으로 메시지 전송. 성공 시 True, 미설정/실패 시 False.

    quant_trader notifier.send_telegram과 동일 패턴 — 동일 env var(TELEGRAM_BOT_TOKEN/
    TELEGRAM_CHAT_ID)를 그대로 읽는다(같은 봇/채팅방 재사용, AC12). 미설정이면 조용히 스킵.
    """
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        logger.warning("텔레그램 토큰 또는 채팅 ID가 설정되지 않았습니다.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message, "parse_mode": "HTML"}
    try:
        resp = requests.post(url, data=data, timeout=10)
        if resp.status_code == 200:
            logger.info("올웨더 텔레그램 알림 전송 성공")
            return True
        logger.error("텔레그램 전송 실패: %s %s", resp.status_code, resp.text)
        return False
    except Exception as e:  # noqa: BLE001 — 알림 실패가 배치 본작업을 막으면 안 됨
        logger.error("텔레그램 전송 오류: %s", e)
        return False
