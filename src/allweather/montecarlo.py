"""몬테카를로 비중 최적화 (Approach B — quant_trader 계산 로직 복제).

.omc/specs/brainstorming-all-weather-portfolio.md AC4/AC5 참고.

이 함수는 quant_trader/portfolio/rebalancer.py::run_monte_carlo 의 계산 로직을 **그대로 복제**한
것이다(직접 import·모듈 공유는 하지 않는다 — quant_trader는 실거래 프로덕션 봇이라 read-only
참고만 함). 벡터화 재작성(Approach A)이 아니라, 이미 실전 검증된 루프 기반 계산을 동일하게 둔다.

quant_trader 원본과의 유일한 차이(스펙에서 확정):
  1) RISK_FREE_RATE(원본 고정 0.045) → 인자 risk_free_rate 로 분리. walk-forward가 리밸런싱
     시점마다 그 시점의 ^IRX 값을 넣어준다(AC6/AC7).
  2) N_SIMULATIONS/seed 를 인자로 노출 — 기본값은 원본과 동일(100,000, 월별 시드 고정).
     walk-forward는 시점별 결정론적 시드를 주입하고, 테스트는 작은 횟수로 빠르게 검증한다.
계산식(연율화 ×252/√252, sharpe=(ret-rf)/vol, argmax sharpe)은 원본과 동일하다.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

# quant_trader rebalancer.py 와 동일한 시뮬레이션 횟수(AC5).
N_SIMULATIONS = 100_000


def run_monte_carlo(
    prices: pd.DataFrame,
    risk_free_rate: float,
    n_simulations: int = N_SIMULATIONS,
    seed: int | None = None,
) -> dict:
    """N회 몬테카를로 시뮬레이션으로 최대 샤프비율 포트폴리오를 산출한다.

    quant_trader run_monte_carlo 로직 복제. risk_free_rate 만 인자로 분리(시점별 ^IRX).

    반환:
      weights      : {ticker: 최적 비중}
      annual_return: 예상 연간 수익률
      annual_vol   : 예상 연간 변동성
      sharpe       : 샤프비율
      corr         : 상관계수 행렬 (dict)
      period_start/period_end : 사용한 데이터 구간
    """
    returns = prices.pct_change().dropna()
    mean_ret = returns.mean()
    cov_matrix = returns.cov()
    n_assets = len(returns.columns)
    tickers = list(returns.columns)

    results = np.zeros((3, n_simulations))
    weights_all = np.zeros((n_simulations, n_assets))

    # 원본은 월별 시드 고정(int(now 'YYYYMM')). walk-forward 재현성/테스트를 위해 seed 주입을
    # 허용하되, 미주입 시 원본과 동일하게 동작한다(그대로 복제 원칙).
    if seed is not None:
        np.random.seed(seed)
    else:
        np.random.seed(int(datetime.now().strftime("%Y%m")))

    for i in range(n_simulations):
        w = np.random.random(n_assets)
        w = w / w.sum()
        weights_all[i] = w

        port_ret = float(np.sum(mean_ret * w) * 252)
        port_vol = float(np.sqrt(np.dot(w.T, np.dot(cov_matrix * 252, w))))
        sharpe = (port_ret - risk_free_rate) / port_vol if port_vol > 0 else 0

        results[0, i] = port_ret
        results[1, i] = port_vol
        results[2, i] = sharpe

    best_idx = results[2].argmax()
    best_w = weights_all[best_idx]

    weights = {t: round(float(w), 4) for t, w in zip(tickers, best_w)}
    corr = returns.corr().round(3).to_dict()

    return {
        "weights": weights,
        "annual_return": round(float(results[0, best_idx]), 4),
        "annual_vol": round(float(results[1, best_idx]), 4),
        "sharpe": round(float(results[2, best_idx]), 4),
        "corr": corr,
        "period_start": str(prices.index[0].date()),
        "period_end": str(prices.index[-1].date()),
    }
