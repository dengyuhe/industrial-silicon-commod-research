"""
Time-series diagnostics implemented with numpy/scipy (no statsmodels dependency).
Results match statsmodels' conventions; tests/test_against_statsmodels.py checks this when
statsmodels is installed.
"""
import numpy as np
from scipy import stats

# MacKinnon (1994) response-surface coefficients for the ADF test with a constant
# (same values as statsmodels.tsa.adfvalues, regression="c", one series)
_TAU_MAX, _TAU_MIN, _TAU_STAR = 2.74, -18.83, -1.61
_SMALL_P = [2.1659, 1.4412, 0.038269]
_LARGE_P = [1.7339, 0.93202, -0.12745, -0.010368]
ADF_CRITICAL = {"1%": -3.43, "5%": -2.86, "10%": -2.57}     # large-sample, constant only


def _ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta, resid


def mackinnon_p(tau):
    if tau > _TAU_MAX:
        return 1.0
    if tau < _TAU_MIN:
        return 0.0
    coefs = _SMALL_P if tau <= _TAU_STAR else _LARGE_P
    return float(stats.norm.cdf(np.polyval(coefs[::-1], tau)))


def adf(x, max_lag=None):
    """
    Augmented Dickey-Fuller test with a constant; lag length chosen by AIC (as statsmodels autolag).
    Returns dict(stat, pvalue, lags, nobs, critical).
    """
    x = np.asarray(x, float)
    n = len(x)
    if max_lag is None:
        max_lag = int(np.ceil(12 * (n / 100) ** 0.25))
    dx = np.diff(x)

    def design(k, start):
        y = dx[start:]
        cols = [np.ones(len(y)), x[start:-1]]
        cols += [dx[start - i:-i] for i in range(1, k + 1)]
        return y, np.column_stack(cols)

    best = None
    for k in range(max_lag + 1):                 # common sample for AIC comparison
        y, X = design(k, max_lag)
        _, e = _ols(y, X)
        aic = len(y) * np.log(e @ e / len(y)) + 2 * X.shape[1]
        if best is None or aic < best[0]:
            best = (aic, k)
    k = best[1]
    y, X = design(k, k)                         # re-estimate on the full sample for that lag
    beta, e = _ols(y, X)
    s2 = e @ e / (len(y) - X.shape[1])
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    tau = beta[1] / se
    return dict(stat=float(tau), pvalue=mackinnon_p(tau), lags=k, nobs=len(y), critical=ADF_CRITICAL)


def acf(x, nlags=30):
    x = np.asarray(x, float) - np.mean(x)
    denom = x @ x
    return np.array([1.0] + [x[:-k] @ x[k:] / denom for k in range(1, nlags + 1)])


def pacf(x, nlags=30):
    """Partial autocorrelations by the Durbin-Levinson recursion on the sample ACF."""
    r = acf(x, nlags)
    out, phi = [1.0], np.zeros(0)
    for k in range(1, nlags + 1):
        if k == 1:
            phi_kk = r[1]
            phi = np.array([phi_kk])
        else:
            phi_kk = (r[k] - phi @ r[1:k][::-1]) / (1 - phi @ r[1:k])
            phi = np.append(phi - phi_kk * phi[::-1], phi_kk)
        out.append(phi_kk)
    return np.array(out)


def ljung_box(resid, lags=(1, 5, 10), dof=0):
    """Ljung-Box Q statistics; returns {lag: (Q, pvalue)}. ``dof`` = number of ARMA params."""
    e = np.asarray(resid, float)
    n = len(e)
    r = acf(e, max(lags))
    out = {}
    for h in lags:
        q = n * (n + 2) * np.sum(r[1:h + 1] ** 2 / (n - np.arange(1, h + 1)))
        out[h] = (float(q), float(stats.chi2.sf(q, max(h - dof, 1))))
    return out


def arch_lm(resid, nlags=10):
    """Engle's ARCH-LM test: regress e_t^2 on its own lags. Returns (LM, pvalue)."""
    e2 = np.asarray(resid, float) ** 2
    y = e2[nlags:]
    X = np.column_stack([np.ones(len(y))] + [e2[nlags - i:-i] for i in range(1, nlags + 1)])
    _, u = _ols(y, X)
    r2 = 1 - (u @ u) / np.sum((y - y.mean()) ** 2)
    lm = len(y) * r2
    return float(lm), float(stats.chi2.sf(lm, nlags))


def jarque_bera(x):
    """Returns (JB, pvalue, skew, kurtosis) with kurtosis on the non-excess scale (normal = 3)."""
    x = np.asarray(x, float)
    jb, p = stats.jarque_bera(x)
    return float(jb), float(p), float(stats.skew(x)), float(stats.kurtosis(x, fisher=False))
