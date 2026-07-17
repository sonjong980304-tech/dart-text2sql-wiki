"""몬테카를로 비중 최적화 — 종목당 최소10%~최대45% 상하한 (AC4/AC5 + 상하한 결정).

.omc/specs/brainstorming-all-weather-portfolio.md AC4/AC5 참고.

원래 이 함수는 quant_trader/portfolio/rebalancer.py::run_monte_carlo 의 계산 로직을 그대로
복제한 것이었다(무제약 샤프비율 극대화). 그런데 2026-07-17 실제 21.7년 데이터로 검증해보니,
무제약 방식이 특정 구간(예: 최근 10년 lookback)에서 QQQ+삼성전자에 99% 가까이 쏠리는 코너
솔루션으로 수렴하는 것을 확인했다 — TLT/금현물처럼 그 구간 수익률이 낮거나 마이너스인 자산은
0%에 가깝게 밀려나, "올웨더(전천후)"라는 취지와 어긋났다. 그래서 종목당 최소10%~최대45% 비중
상하한을 추가했다(사용자 결정, quant_trader 원본과의 실질적 차이는 이 상하한 하나뿐 — 연율화
(×252/√252)·샤프비율 공식(sharpe=(ret-rf)/vol)·argmax 방식은 원본과 동일).

quant_trader 원본과의 차이:
  1) RISK_FREE_RATE(원본 고정 0.045) → 인자 risk_free_rate 로 분리(AC6/AC7, walk-forward가
     리밸런싱 시점마다 ^IRX 값을 넣어준다).
  2) N_SIMULATIONS/seed 를 인자로 노출 — 기본값은 원본과 동일(100,000).
  3) [신규] 종목당 비중 상하한(MIN_WEIGHT~MAX_WEIGHT) — 원본엔 없던 제약. 거부샘플링(rejection
     sampling)으로 상하한을 만족하는 조합만 후보로 남긴다.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

# quant_trader rebalancer.py 와 동일한 시뮬레이션 횟수(AC5).
N_SIMULATIONS = 100_000

# 종목당 비중 상하한 — 무제약 샤프비율 극대화가 코너 솔루션(한두 종목 몰빵)으로 수렴하는 것을
# 막기 위한 신규 제약(2026-07-17 실측 검증 후 결정). 4종목 기준 10%*4=40%<=1.0 이라 실현 가능.
MIN_WEIGHT = 0.10
MAX_WEIGHT = 0.45


def _sample_bounded_weights(n_assets: int, n_simulations: int, rng: np.random.Generator) -> np.ndarray:
    """상하한(MIN_WEIGHT~MAX_WEIGHT)을 만족하는 비중 조합을 n_simulations개 뽑는다.

    Dirichlet(1,...,1)은 단체(simplex) 위 균등분포라 원본의 uniform-정규화 방식보다 편향이 적다.
    거부샘플링: 상하한을 만족하는 것만 남기고, 부족하면 더 뽑는다(최대 10회 시도).
    상하한 자체가 실현 불가능한 조합이면(자산 수 대비 모순) 균등비중 1개로 폴백한다.
    """
    accepted: list[np.ndarray] = []
    total = 0
    for _ in range(10):
        if sum(len(a) for a in accepted) >= n_simulations:
            break
        batch = n_simulations * 3
        candidates = rng.dirichlet(np.ones(n_assets), size=batch)
        mask = (candidates.min(axis=1) >= MIN_WEIGHT) & (candidates.max(axis=1) <= MAX_WEIGHT)
        accepted.append(candidates[mask])
        total += batch

    pool = np.concatenate(accepted) if accepted else np.empty((0, n_assets))
    if len(pool) == 0:
        return np.full((1, n_assets), 1.0 / n_assets)
    return pool[:n_simulations]


def run_monte_carlo(
    prices: pd.DataFrame,
    risk_free_rate: float,
    n_simulations: int = N_SIMULATIONS,
    seed: int | None = None,
) -> dict:
    """상하한(MIN_WEIGHT~MAX_WEIGHT) 안에서 N회 몬테카를로로 최대 샤프비율 포트폴리오를 산출한다.

    반환:
      weights      : {ticker: 최적 비중} (모두 MIN_WEIGHT~MAX_WEIGHT 이내)
      annual_return: 예상 연간 수익률
      annual_vol   : 예상 연간 변동성
      sharpe       : 샤프비율
      corr         : 상관계수 행렬 (dict)
      period_start/period_end : 사용한 데이터 구간
    """
    returns = prices.pct_change().dropna()
    mean_ret = returns.mean().values
    cov_matrix = returns.cov().values
    n_assets = len(returns.columns)
    tickers = list(returns.columns)

    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng(int(datetime.now().strftime("%Y%m")))

    W = _sample_bounded_weights(n_assets, n_simulations, rng)

    # macOS Accelerate BLAS의 알려진 부작용으로 대량 matmul에서 divide-by-zero/overflow/invalid
    # RuntimeWarning이 허위로 뜰 수 있다(실측 확인: W/mean_ret에 NaN 없고 결과값도 정상 범위 —
    # SIMD 연산 중 미사용 메모리 레인을 스치면서 나는 경고일 뿐, 실제 계산 오류 아님).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        port_ret = (W @ mean_ret) * 252
        ann_cov = cov_matrix * 252
        port_vol = np.sqrt(np.einsum("ij,jk,ik->i", W, ann_cov, W))
    sharpe = np.where(port_vol > 0, (port_ret - risk_free_rate) / port_vol, 0.0)

    best_idx = int(sharpe.argmax())
    best_w = W[best_idx]

    weights = {t: round(float(w), 4) for t, w in zip(tickers, best_w)}
    corr = returns.corr().round(3).to_dict()

    return {
        "weights": weights,
        "annual_return": round(float(port_ret[best_idx]), 4),
        "annual_vol": round(float(port_vol[best_idx]), 4),
        "sharpe": round(float(sharpe[best_idx]), 4),
        "corr": corr,
        "period_start": str(prices.index[0].date()),
        "period_end": str(prices.index[-1].date()),
    }
