"""코스피 지수(KOSPI) 실제 시세 최초 적재 스크립트.

네이버 fchart는 개별 종목과 동일한 심볼 규약으로 지수도 제공한다
(symbol=KOSPI, https://fchart.stock.naver.com/sise.nhn?symbol=KOSPI&...). 개별종목 수집
로직(ingest_naver_prices)을 그대로 재사용해 prices 테이블에 stock_code='KOSPI'로 적재한다
(company 테이블엔 등록하지 않는 가상 종목코드 — src/eval/factcheck/backtest.py가 이미
이 코드를 전제하고 있었다). count=3000은 기존 개별종목 백필과 동일한 기준(약 12년치)이라
백테스트 실제 커버 범위와 맞춰둔다.

실행: python3 scripts/backfill_kospi_index.py
이후 일일 갱신은 scripts/run_naver_prices.py가 codes 목록에 "KOSPI"를 포함해 자동으로 이어받는다.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingest.naver_prices import ingest_naver_prices

if __name__ == "__main__":
    result = ingest_naver_prices(codes=["KOSPI"], count=3000)
    print(result)
