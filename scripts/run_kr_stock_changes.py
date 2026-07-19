"""KRX 기업 주요 변동이력(상호/업종/액면 변경) 전종목 수집 실행 스크립트 (백필).

pykrx get_stock_major_changes(ticker) 는 종목별로 1975년부터의 상호/업종/액면 변경 이력을
시점조회 없이 통째로 반환한다(src/ingest/kr_stock_changes.py 참고). kr_trading_status(매일
스냅샷 diff)와 달리 과거 이력이 한 번에 오므로 한 차례 전종목 순회로 전체 이력을 적재한다.
이후에는 신규 사명 변경 반영을 위해 가끔(예: 월 1회) 재실행하면 된다(UNIQUE upsert 로 멱등).

전종목 순회는 종목별 실패를 격리(스킵+Slack 알림)하고 call_with_retry 로 재시도한다. 라이브
pykrx 호출은 이 스크립트를 실행할 때만 일어난다. 기존 run_kr_trading_status.py 와 동일한
명명·구조 관례. 실행: python3 scripts/run_kr_stock_changes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.kr_stock_changes import ingest_stock_changes

if __name__ == "__main__":
    result = ingest_stock_changes()
    print(
        f"[kr_stock_changes] tickers={result['tickers']}/{result['total_codes']} "
        f"rows_stored={result['rows_stored']} failed={len(result['failed'])}"
    )
    if result["failed"]:
        print(f"[kr_stock_changes] 실패 종목: {result['failed']}")
