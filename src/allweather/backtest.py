"""올웨더 walk-forward 백테스트 엔진.

.omc/specs/brainstorming-all-weather-portfolio.md AC3/AC8/AC14 참고.

매 리밸런싱 시점(월간, 약 120회)마다 그 시점까지의 데이터로만 몬테카를로를 재계산해 최적 비중을
구하고(look-ahead 없음, AC3), 그 비중으로 다음 리밸런싱 시점까지 보유했을 때의 실현 수익률을
누적해 자산곡선(NAV)을 만든다. 곡선에서 MDD/CAGR/누적수익률/샤프비율을 산출한다(AC14).

몬테카를로(10만회)는 배치에서만 실제로 돌고, 테스트/재사용을 위해 monte_carlo_fn을 주입 가능하게
둔다. 무위험이자율은 리밸런싱 시점별 ^IRX 과거값을 쓴다(risk_free_rate_at, AC7).
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from .data import risk_free_rate_at
from .montecarlo import N_SIMULATIONS, run_monte_carlo


# ---------------------------------------------------------------------------
# 리밸런싱 시점 — 월별 마지막 거래일 (실제 존재하는 거래일만, 약 120개/10년)
# ---------------------------------------------------------------------------
def rebalance_points(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """가격 인덱스에서 각 (연,월)의 마지막 거래일을 리밸런싱 시점으로 뽑는다(오름차순)."""
    if len(index) == 0:
        return []
    s = pd.Series(index, index=index)
    last = s.groupby([index.year, index.month]).max()
    return sorted(last.tolist())


# ---------------------------------------------------------------------------
# 곡선 지표
# ---------------------------------------------------------------------------
def max_drawdown(navs: list[float]) -> float:
    """자산곡선(NAV) 리스트의 최대낙폭(MDD, 0 이하 소수). 하락이 없으면 0."""
    peak = navs[0]
    mdd = 0.0
    for v in navs:
        if v > peak:
            peak = v
        dd = v / peak - 1.0
        if dd < mdd:
            mdd = dd
    return mdd


def cagr(navs: list[float], years: float) -> float:
    """연평균 복리 수익률(CAGR). 기간이 0 이하이거나 시작값이 0 이하이면 0."""
    if years <= 0 or navs[0] <= 0:
        return 0.0
    return (navs[-1] / navs[0]) ** (1.0 / years) - 1.0


def sharpe_ratio(period_returns: list[float], avg_risk_free: float, periods_per_year: int = 12) -> float:
    """실현 구간수익률에서 연율화 샤프비율. std=0이면 0(무위험 초과 계산 불가)."""
    if len(period_returns) < 2:
        return 0.0
    s = pd.Series(period_returns)
    std = float(s.std(ddof=1))
    if std <= 0:
        return 0.0
    ann_ret = float(s.mean()) * periods_per_year
    ann_vol = std * (periods_per_year ** 0.5)
    return (ann_ret - avg_risk_free) / ann_vol


# ---------------------------------------------------------------------------
# walk-forward 백테스트
# ---------------------------------------------------------------------------
def run_walk_forward(
    panel: pd.DataFrame,
    irx_series: pd.Series,
    *,
    monte_carlo_fn=run_monte_carlo,
    n_simulations: int = N_SIMULATIONS,
    lookback_years: int = 10,
    today: date | None = None,
) -> dict:
    """월간 walk-forward 백테스트를 실행해 스냅샷 dict를 반환한다.

    각 리밸런싱 시점 t: [t-lookback_years, t] 구간(있는 만큼만, look-ahead 없음)으로 몬테카를로를
    돌려 최적 비중을 구하고, t→다음 시점 구간의 실현 수익률을 누적한다. 마지막 시점의 비중이
    '현재 추천 비중'(화면·알림에 노출).

    반환: computed_at / weights / cagr / mdd / sharpe / cumulative_return / backtest_curve.
    """
    today = today or date.today()
    cols = list(panel.columns)
    points = rebalance_points(panel.index)

    nav = 1.0
    curve: list[dict] = []
    period_returns: list[float] = []
    rfs: list[float] = []
    latest_weights: dict[str, float] = {c: 1.0 / len(cols) for c in cols}

    started = False
    for i, t in enumerate(points):
        lo = t - pd.DateOffset(years=lookback_years)
        window = panel[(panel.index <= t) & (panel.index >= lo)]
        if len(window) < 2:
            continue  # 데이터가 2행 미만이면 아직 계산 불가(있는 만큼만 — 예외 없이 스킵)

        rf = risk_free_rate_at(irx_series, t)
        rfs.append(rf)
        # 시점별 결정론적 시드(재현성) — YYYYMMDD 정수.
        seed = int(pd.Timestamp(t).strftime("%Y%m%d"))
        res = monte_carlo_fn(window, risk_free_rate=rf, n_simulations=n_simulations, seed=seed)
        w = res["weights"]
        latest_weights = w

        if not started:
            curve.append({"date": str(pd.Timestamp(t).date()), "nav": round(nav, 6)})
            started = True

        # t → 다음 리밸런싱 시점까지 이 비중으로 보유했을 때의 실현 수익률.
        if i < len(points) - 1:
            t_next = points[i + 1]
            p0 = panel.loc[t]
            p1 = panel.loc[t_next]
            port_ret = 0.0
            for c in cols:
                if p0[c] and p0[c] > 0:
                    port_ret += w.get(c, 0.0) * (p1[c] / p0[c] - 1.0)
            nav *= (1.0 + port_ret)
            period_returns.append(port_ret)
            curve.append({"date": str(pd.Timestamp(t_next).date()), "nav": round(nav, 6)})

    navs = [pt["nav"] for pt in curve] or [1.0]
    if len(curve) >= 2:
        d0 = pd.Timestamp(curve[0]["date"])
        d1 = pd.Timestamp(curve[-1]["date"])
        years = (d1 - d0).days / 365.25
    else:
        years = 0.0
    avg_rf = sum(rfs) / len(rfs) if rfs else 0.0

    return {
        "computed_at": today.strftime("%Y-%m-%d"),
        "weights": latest_weights,
        "cagr": round(cagr(navs, years), 6),
        "mdd": round(max_drawdown(navs), 6),
        "sharpe": round(sharpe_ratio(period_returns, avg_rf), 6),
        "cumulative_return": round(navs[-1] / navs[0] - 1.0, 6),
        "backtest_curve": curve,
    }
