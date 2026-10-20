"""
Cross-check the numpy/scipy implementations against statsmodels (skipped if not installed).
    pip install statsmodels && python -m pytest tests/test_against_statsmodels.py
"""
import numpy as np
import pytest

sm = pytest.importorskip("statsmodels.api")
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch  # noqa: E402
from statsmodels.tsa.arima.model import ARIMA  # noqa: E402
from statsmodels.tsa.stattools import adfuller, pacf as sm_pacf  # noqa: E402

from sifutures import arima, timeseries as ts  # noqa: E402

rng = np.random.default_rng(7)
e = rng.normal(0, 10, 1501)
y = 1000 + np.cumsum(e[1:] - 0.3 * e[:-1])


def test_adf_matches():
    ours = ts.adf(y)
    stat, p, lags, *_ = adfuller(y, regression="c", autolag="AIC")
    assert ours["lags"] == lags
    assert ours["stat"] == pytest.approx(stat, abs=1e-6)
    assert ours["pvalue"] == pytest.approx(p, abs=1e-4)


def test_ljung_box_and_arch_match():
    w = np.diff(y)
    lb = acorr_ljungbox(w, lags=[10])
    assert ts.ljung_box(w, (10,))[10][0] == pytest.approx(float(lb["lb_stat"].iloc[0]), rel=1e-8)
    lm, lmp, *_ = het_arch(w - w.mean(), nlags=5)
    assert ts.arch_lm(w - w.mean(), 5)[0] == pytest.approx(lm, rel=1e-6)


def test_pacf_matches_durbin_levinson():
    w = np.diff(y)
    assert np.allclose(ts.pacf(w, 10), sm_pacf(w, nlags=10, method="ldbiased"), atol=1e-8)


def test_arima_close_to_exact_mle():
    ours = arima.fit(y, (0, 1, 1)).params["ma.L1"]
    theirs = ARIMA(y, order=(0, 1, 1)).fit().params[0]
    assert ours == pytest.approx(theirs, abs=0.01)


def test_garch_t_close_to_arch_package():
    """Cross-check GARCH(1,1)-t against the `arch` package (skipped if not installed)."""
    arch = pytest.importorskip("arch")
    from sifutures import volatility as V
    g = np.random.default_rng(11)
    n, e, h = 3000, np.zeros(3000), 1.0
    for t in range(n):
        e[t] = np.sqrt(h) * g.standard_t(6) * np.sqrt(4 / 6)
        h = 0.05 + 0.08 * e[t] ** 2 + 0.88 * h
    ours = V.fit(e, "garch", "t").params
    ref = arch.arch_model(e, mean="Constant", vol="GARCH", p=1, q=1, dist="t").fit(disp="off").params
    assert ours["alpha"] == pytest.approx(ref["alpha[1]"], abs=0.02)
    assert ours["beta"] == pytest.approx(ref["beta[1]"], abs=0.03)
    assert ours["nu"] == pytest.approx(ref["nu"], abs=1.0)
