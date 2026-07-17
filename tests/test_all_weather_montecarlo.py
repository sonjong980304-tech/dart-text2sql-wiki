"""올웨더 몬테카를로 비중 최적화 테스트 (AC4/AC5 + 종목당 비중 상하한).

.omc/specs/brainstorming-all-weather-portfolio.md 참고.
quant_trader/portfolio/rebalancer.py의 몬테카를로 계산식(연율화, sharpe 공식)은 그대로 따르되,
2026-07-17 21.7년 실측 검증 결과 무제약 샤프비율 극대화가 한두 종목(QQQ+삼성전자)에 99% 가까이
쏠리는 코너 솔루션으로 수렴하는 것을 확인해, 종목당 최소10%~최대45% 비중 상하한을 추가했다
(사용자 결정 — quant_trader 원본과의 유일한 실질적 차이). 계산은 순수 로직(DB/네트워크 의존
없음)이라 작은 시뮬레이션 횟수로 빠르게 검증하고, N_SIMULATIONS 상수(100,000)만 별도로
고정한다(실제 배치는 이 상수를 쓴다).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.allweather.montecarlo import MAX_WEIGHT, MIN_WEIGHT, N_SIMULATIONS, run_monte_carlo


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


def test_weights_respect_min_and_max_bounds():
    # 21.7년 실측에서 무제약 방식이 한두 종목에 99% 가까이 쏠리는 것을 확인해 상하한을 추가했다.
    p = _panel()
    res = run_monte_carlo(p, risk_free_rate=0.045, n_simulations=3000, seed=5)
    for w in res["weights"].values():
        assert MIN_WEIGHT - 1e-9 <= w <= MAX_WEIGHT + 1e-9


def test_bounds_prevent_near_zero_concentration_even_when_one_asset_dominates():
    # 한 종목이 압도적으로 좋고 다른 종목이 마이너스여도(TLT가 2022년에 그랬듯) 상하한 때문에
    # 그 종목이 0%까지 밀리지는 않는다 — "안전자산은 성과가 나빠도 최소한은 들고 간다"는 취지.
    dates = pd.date_range("2020-01-01", periods=200, freq="B")
    rng = np.random.default_rng(1)
    data = {
        "QQQ": 100 * np.exp(np.cumsum(rng.normal(0.003, 0.01, size=200))),
        "005930.KS": 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, size=200))),
        "TLT": 100 * np.exp(np.cumsum(rng.normal(-0.001, 0.008, size=200))),
        "411060.KS": 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.012, size=200))),
    }
    p = pd.DataFrame(data, index=dates)
    res = run_monte_carlo(p, risk_free_rate=0.02, n_simulations=5000, seed=3)
    for w in res["weights"].values():
        assert w >= MIN_WEIGHT - 1e-9


def test_min_weight_times_asset_count_is_feasible():
    # 상하한 자체가 모순이면(예: 종목수*최소비중 > 1) 어떤 조합도 못 뽑는다 — 지금 4종목 기준으로
    # 10%*4=40% <= 1.0 이라 실현 가능해야 한다는 전제를 고정해둔다(회귀 방지).
    assert MIN_WEIGHT * 4 <= 1.0
    assert MAX_WEIGHT * 4 >= 1.0
