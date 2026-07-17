"""올웨더 몬테카를로 비중 최적화 테스트 (AC4/AC5).

.omc/specs/brainstorming-all-weather-portfolio.md 참고.
quant_trader/portfolio/rebalancer.py의 몬테카를로 계산 로직을 그대로 복제한 함수(Approach B)를
검증한다. 계산은 순수 로직(DB/네트워크 의존 없음)이라 작은 시뮬레이션 횟수로 빠르게 검증하고,
N_SIMULATIONS 상수(100,000)만 별도로 고정한다(실제 배치는 이 상수를 쓴다).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.allweather.montecarlo import N_SIMULATIONS, run_monte_carlo


def _panel(n: int = 60) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    data = {}
    for i, c in enumerate(["QQQ", "005930.KS", "TLT", "411060.KS"]):
        steps = rng.normal(0.0005 * (i + 1), 0.01, size=n)
        data[c] = 100 * np.exp(np.cumsum(steps))
    return pd.DataFrame(data, index=dates)


def test_n_simulations_constant_is_100k():
    # AC5: 시뮬레이션 횟수 상수는 quant_trader와 동일한 100,000.
    assert N_SIMULATIONS == 100_000


def test_run_monte_carlo_returns_normalized_weights_over_all_tickers():
    p = _panel()
    res = run_monte_carlo(p, risk_free_rate=0.045, n_simulations=2000, seed=42)
    assert set(res["weights"].keys()) == set(p.columns)
    assert abs(sum(res["weights"].values()) - 1.0) < 1e-3
    for w in res["weights"].values():
        assert 0.0 <= w <= 1.0


def test_sharpe_reflects_risk_free_rate():
    # sharpe = (annual_return - rf)/annual_vol 를 재구성해 rf가 실제 계산에 쓰였는지 확인(AC6/AC7 근거).
    p = _panel()
    res = run_monte_carlo(p, risk_free_rate=0.045, n_simulations=3000, seed=7)
    recon = res["sharpe"] * res["annual_vol"] + 0.045
    assert abs(recon - res["annual_return"]) < 1e-2


def test_higher_risk_free_rate_lowers_max_sharpe_for_same_seed():
    # 같은 시드(동일 후보 포트폴리오)면 무위험이자율이 높을수록 최대 샤프비율은 낮아진다.
    p = _panel()
    lo = run_monte_carlo(p, risk_free_rate=0.00, n_simulations=3000, seed=11)
    hi = run_monte_carlo(p, risk_free_rate=0.20, n_simulations=3000, seed=11)
    assert hi["sharpe"] < lo["sharpe"]
