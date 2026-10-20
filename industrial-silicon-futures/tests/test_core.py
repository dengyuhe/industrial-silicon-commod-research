"""Tests on simulated data (no vendor data needed). Run: python -m pytest tests"""
import numpy as np
import pandas as pd
import pytest

from sifutures import arima, data as D, evaluate as ev, fundamentals as F, timeseries as ts

rng = np.random.default_rng(1)


def simulate_ma1(theta, n=3000, sigma=10.0, drift=0.0):
    e = rng.normal(0, sigma, n + 1)
    return 1000 + np.cumsum(drift + e[1:] + theta * e[:-1])


def test_arima_recovers_ma_parameter():
    r = arima.fit(simulate_ma1(-0.4), (0, 1, 1))
    assert abs(r.params["ma.L1"] + 0.4) < 0.05
    assert abs(r.sigma2 / 100 - 1) < 0.1


def test_random_walk_forecast_interval_grows_with_sqrt_h():
    y = 1000 + np.cumsum(rng.normal(0, 5, 2000))
    r = arima.fit(y, (0, 1, 0))
    mean, lo, hi = r.forecast(25)
    assert np.allclose(mean, y[-1])
    width = (hi - lo) / 2
    assert np.isclose(width[24] / width[0], 5.0, rtol=1e-6)


def test_adf_distinguishes_unit_root_from_stationary():
    assert ts.adf(1000 + np.cumsum(rng.normal(size=800)))["pvalue"] > 0.05
    assert ts.adf(rng.normal(size=800))["pvalue"] < 0.01
    assert abs(ts.mackinnon_p(-2.86) - 0.05) < 0.002


def test_arch_lm_detects_volatility_clustering():
    n, w, a0, a1 = 3000, np.zeros(3000), 0.1, 0.6
    for t in range(1, n):
        w[t] = rng.normal() * np.sqrt(a0 + a1 * w[t - 1] ** 2)
    assert ts.arch_lm(w, 5)[1] < 0.001
    assert ts.arch_lm(rng.normal(size=3000), 5)[1] > 0.01


def test_pacf_of_ar1_cuts_off_after_lag_one():
    x = np.zeros(5000)
    for t in range(1, 5000):
        x[t] = 0.6 * x[t - 1] + rng.normal()
    p = ts.pacf(x, 5)
    assert abs(p[1] - 0.6) < 0.05 and np.all(np.abs(p[2:]) < 0.05)


def test_diebold_mariano_sign():
    a = rng.normal(size=500)
    good, bad = a + rng.normal(0, 0.1, 500), a + rng.normal(0, 1, 500)
    stat, p = ev.diebold_mariano(a, good, bad)
    assert stat < 0 and p < 0.001


def test_roll_detection_and_adjustment():
    idx = pd.bdate_range("2024-01-01", periods=6)
    near = pd.Series([100, 101, 102, 101, 100, 99.0], idx)
    second = pd.Series([130, 131, 132, 131.5, 130, 129.0], idx)
    active = pd.concat([near.iloc[:3], second.iloc[3:]])            # switches on day 4
    gaps = D.roll_gaps(active, pd.DataFrame({"nearby1_close": near, "nearby2_close": second}))
    assert list(gaps.index) == [idx[3]]
    adj = D.roll_adjust(active, gaps, "difference")
    assert np.allclose(adj.diff().iloc[1:].values, [1, 1, -0.5, -1.5, -1])   # second contract's own moves
    adj_r = D.roll_adjust(active, gaps, "ratio")
    assert np.isclose(adj_r.iloc[3] / adj_r.iloc[2], 131.5 / 132)


@pytest.mark.parametrize("scenario, total", [("Hoshine, Shanshan base (2024)", 8300), ("Xinjiang average", 9245),
                                             ("Southwest, wet season", 10460), ("Southwest, dry season", 12620)])
def test_cost_tables_match_report(scenario, total):
    assert F.total_cost(scenario) == pytest.approx(total)


def test_electricity_elasticity_equals_cost_share():
    m = F.electricity_sensitivity()
    numeric = (m.loc[0.30, 11000] / m.loc[0.28, 11000] - 1) / (0.30 / 0.28 - 1)
    assert F.electricity_elasticity() == pytest.approx(numeric, rel=1e-6)
    assert numeric == pytest.approx(0.371, abs=0.001)


def test_silicon_intensity_range_matches_report():
    g = F.silicon_intensity()["g_per_watt"]
    assert 1.55 < g.min() < 1.60 and 2.00 < g.max() < 2.03


# --- Volatility models ---------------------------------------------------------------------------------
from sifutures import volatility as V  # noqa: E402


def simulate_gjr_t(n=4000, omega=0.05, alpha=0.05, gamma=0.08, beta=0.85, nu=6):
    e, h = np.zeros(n), omega / (1 - alpha - gamma / 2 - beta)
    for t in range(n):
        e[t] = np.sqrt(h) * rng.standard_t(nu) * np.sqrt((nu - 2) / nu)
        h = omega + (alpha + gamma * (e[t] < 0)) * e[t] ** 2 + beta * h
    return e


def test_variance_filter_matches_explicit_recursion():
    e = rng.normal(size=60)
    h = np.empty(61)
    h[0] = 1.3
    for t in range(60):
        h[t + 1] = 0.1 + (0.05 + 0.1 * (e[t] < 0)) * e[t] ** 2 + 0.8 * h[t]
    assert np.allclose(V._filter(e, 0.1, 0.05, 0.1, 0.8, 1.3), h)


def test_gjr_garch_t_recovers_parameters_and_wins_on_aic():
    e = simulate_gjr_t()
    fits = {m + d: V.fit(e, m, d) for m in ("garch", "gjr") for d in ("normal", "t")}
    p = fits["gjrt"].params
    assert abs(p["beta"] - 0.85) < 0.05 and abs(p["gamma"] - 0.08) < 0.05 and abs(p["nu"] - 6) < 1.5
    assert min(fits, key=lambda k: fits[k].aic) == "gjrt"


def test_garch_removes_arch_effects():
    e = simulate_gjr_t()
    assert ts.arch_lm(e, 5)[1] < 0.001
    z = V.fit(e, "gjr", "t").std_resid
    assert ts.arch_lm(z, 5)[1] > 0.01


def test_kupiec_accepts_correct_rate_and_rejects_wrong_rate():
    hits = np.zeros(1000, bool)
    hits[::20] = True                                     # exactly 5%
    assert V.kupiec(hits, 0.05)[1] > 0.9
    hits[::10] = True                                     # 10%
    assert V.kupiec(hits, 0.05)[1] < 0.001
