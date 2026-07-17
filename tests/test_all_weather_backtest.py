"""올웨더 walk-forward 백테스트 테스트 (AC3/AC8/AC14 지표).

매 리밸런싱 시점(월간, 약 120회)마다 그 시점까지의 데이터로만 몬테카를로를 재계산하고(look-ahead
없음), 실현 수익률 곡선에서 MDD/CAGR/누적수익률/샤프비율을 계산한다. 실제 몬테카를로(10만회)는
느리므로 주입 가능한 fake 계산 함수로 대체해 오케스트레이션(호출 횟수·시점 절단·지표 산출)만 검증한다.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.allweather.backtest import rebalance_points, run_walk_forward


def _panel(years: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2014-01-01", periods=years * 252, freq="B")
    rng = np.random.default_rng(1)
    data = {}
    for i, c in enumerate(["QQQ", "005930.KS", "TLT", "411060.KS"]):
        steps = rng.normal(0.0003, 0.008, size=len(dates))
        data[c] = 100 * np.exp(np.cumsum(steps))
    return pd.DataFrame(data, index=dates)


def _fake_mc(prices, **kw):
    cols = list(prices.columns)
    return {
        "weights": {c: 1.0 / len(cols) for c in cols},
        "annual_return": 0.1,
        "annual_vol": 0.1,
        "sharpe": 1.0,
        "period_start": str(prices.index[0].date()),
        "period_end": str(prices.index[-1].date()),
    }


def test_walk_forward_calls_monte_carlo_about_120_times():
    # AC8: 10년 월간 walk-forward → 리밸런싱 시점 수(약 120회)만큼 몬테카를로 호출.
    panel = _panel(10)
    calls = []

    def mc(prices, **kw):
        calls.append(prices)
        return _fake_mc(prices, **kw)

    irx = pd.Series([5.0] * len(panel.index), index=panel.index)
    run_walk_forward(panel, irx, monte_carlo_fn=mc, n_simulations=10, today=date(2024, 1, 1))
    n_points = len(rebalance_points(panel.index))
    assert len(calls) == n_points
    assert 110 <= len(calls) <= 130


def test_no_lookahead_window_only_uses_past_data():
    # AC3: 각 리밸런싱 시점의 window 최댓 인덱스가 그 시점 이하 — 미래 데이터가 섞이지 않는다.
    panel = _panel(3)  # 3년 → 트레일링 10년 룩백 미달(있는 만큼만)
    pts = rebalance_points(panel.index)
    seen = []

    def mc(prices, **kw):
        seen.append(prices.index.max())
        return _fake_mc(prices, **kw)

    irx = pd.Series([5.0] * len(panel.index), index=panel.index)
    run_walk_forward(
        panel, irx, monte_carlo_fn=mc, n_simulations=10, today=date(2017, 1, 1), lookback_years=10
    )
    assert len(seen) == len(pts)
    for max_idx, t in zip(seen, pts):
        assert max_idx <= t  # 미래참조 없음


def test_early_period_computes_with_available_data_no_exception():
    # AC3: 룩백 10년 미달 구간에서도 예외 없이 "있는 만큼만"으로 계산된다.
    panel = _panel(3)
    captured = []

    def mc(prices, **kw):
        captured.append(len(prices))
        return _fake_mc(prices, **kw)

    irx = pd.Series([5.0] * len(panel.index), index=panel.index)
    snap = run_walk_forward(
        panel, irx, monte_carlo_fn=mc, n_simulations=10, today=date(2017, 1, 1), lookback_years=10
    )
    # 3년 패널이므로 어떤 window도 10년(약 2520 거래일)에 못 미친다 = 있는 만큼만.
    assert all(0 < c < 10 * 252 for c in captured)
    assert snap["backtest_curve"]  # 곡선이 정상 생성됨


def test_snapshot_has_all_metrics():
    # AC14 근거: 비중/MDD/샤프/누적수익률/CAGR이 모두 산출된다.
    panel = _panel(10)
    irx = pd.Series([5.0] * len(panel.index), index=panel.index)
    snap = run_walk_forward(
        panel, irx, monte_carlo_fn=lambda p, **k: _fake_mc(p, **k), n_simulations=10,
        today=date(2024, 1, 1),
    )
    for key in ("computed_at", "weights", "cagr", "mdd", "sharpe", "cumulative_return", "backtest_curve"):
        assert key in snap
    assert set(snap["weights"].keys()) == set(panel.columns)
    assert isinstance(snap["backtest_curve"], list) and len(snap["backtest_curve"]) >= 2
    assert snap["mdd"] <= 0  # 최대낙폭은 0 이하
    assert snap["computed_at"] == "2024-01-01"
