"""올웨더 텔레그램 알림 델타 계산 테스트 (AC13).

이력 테이블에 직전 달 행이 있으면 알림 메시지에 비중 변경분(델타, 예: "25%→30%(+5%p)")을
포함한다. 실제 전송(requests.post)은 별도 함수라 여기선 메시지 문자열 생성만 검증한다.
"""
from __future__ import annotations

from src.allweather.notify import build_delta_message


def _cur():
    return {
        "weights": {"005930.KS": 0.30, "QQQ": 0.40, "TLT": 0.20, "411060.KS": 0.10},
        "cagr": 0.12,
        "mdd": -0.2,
        "sharpe": 1.1,
        "cumulative_return": 0.9,
    }


def test_delta_message_includes_change_from_previous():
    prev = {"weights": {"005930.KS": 0.25, "QQQ": 0.45, "TLT": 0.20, "411060.KS": 0.10}}
    msg = build_delta_message(_cur(), prev)
    assert "삼성전자" in msg
    assert "25" in msg and "30" in msg  # 직전 → 이번 비중
    assert "+5" in msg  # +5%p 상승
    assert "%p" in msg


def test_delta_message_without_previous_shows_current_only():
    msg = build_delta_message(_cur(), None)
    assert "삼성전자" in msg
    assert "30" in msg
    # 직전 행이 없으면 델타 표기(%p)는 없어도 된다(첫 실행).


def test_delta_message_includes_performance_metrics():
    msg = build_delta_message(_cur(), None)
    # 화면과 동일 지표를 알림에도 요약(CAGR/MDD/샤프 등) — 최소 하나 이상 포함.
    assert "CAGR" in msg or "샤프" in msg or "MDD" in msg
