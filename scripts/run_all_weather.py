"""올웨더 포트폴리오 배치 실행 스크립트 (launchd용, 매달 1일).

4종목(QQQ/삼성전자/TLT/ACE KRX금현물) 가격으로 walk-forward 백테스트(매 리밸런싱 시점마다
그 시점까지 데이터로 몬테카를로 10만회 재계산, look-ahead 없음)를 돌려 비중/MDD/CAGR/누적수익률/
샤프비율을 계산해 all_weather_snapshot에 이력으로 저장하고, quant_trader와 동일한 텔레그램 채널로
직전 달 대비 비중 변경분(델타)을 포함한 알림을 보낸다. 실거래 주문은 하지 않는다(표시+알림 전용).

실행: python3 scripts/run_all_weather.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.allweather.pipeline import run_all_weather_pipeline

if __name__ == "__main__":
    result = run_all_weather_pipeline()
    snap = result["snapshot"]
    print(
        "[all_weather] computed_at={ca} weights={w} cagr={cagr} mdd={mdd} "
        "sharpe={sharpe} cumret={cum} alerted={al}".format(
            ca=snap["computed_at"], w=snap["weights"], cagr=snap["cagr"], mdd=snap["mdd"],
            sharpe=snap["sharpe"], cum=snap["cumulative_return"], al=result["alerted"],
        )
    )
