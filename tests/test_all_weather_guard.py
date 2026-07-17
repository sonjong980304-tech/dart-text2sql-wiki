"""올웨더 배치 정적 가드 테스트 (AC9).

이 기능은 계산 결과의 저장·표시·알림까지만 담당하고 실거래 주문(매수/매도)을 실행하지 않는다.
배치 관련 소스 어디에도 주문 실행 API 호출 흔적이 없어야 한다(설계 의도 정적 확인).
"""
from __future__ import annotations

from pathlib import Path

# quant_trader 실거래 인프라(KIS 주문)에서 유래할 법한 주문 실행 흔적 목록.
_FORBIDDEN = [
    "order_cash",
    "KISTrader",
    "send_order",
    "place_order",
    "buy_order",
    "sell_order",
    "inquire_balance",
    "매수 주문",
    "매도 주문",
]


def test_no_trade_order_api_in_allweather_sources():
    root = Path(__file__).resolve().parent.parent
    files = list((root / "src" / "allweather").glob("*.py"))
    runner = root / "scripts" / "run_all_weather.py"
    if runner.exists():
        files.append(runner)
    assert files, "올웨더 소스 파일을 찾지 못했습니다."
    for f in files:
        txt = f.read_text(encoding="utf-8")
        for bad in _FORBIDDEN:
            assert bad not in txt, f"{f.name}에 주문 실행 흔적 발견: {bad!r}"
