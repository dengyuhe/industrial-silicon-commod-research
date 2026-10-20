"""
ARIMA(p, 1, q) with optional drift, estimated by conditional sum of squares (CSS).

The model for the differenced series w_t = y_t - y_{t-1} is
    w_t = c + phi_1 w_{t-1} + ... + phi_p w_{t-p} + e_t + theta_1 e_{t-1} + ... + theta_q e_{t-q}
CSS and exact maximum likelihood (statsmodels' default) give near-identical estimates for
samples of this size; tests/test_against_statsmodels.py compares them.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats


def _residuals(w, c, phi, theta):
    p, q, n = len(phi), len(theta), len(w)
    e = np.zeros(n)
    for t in range(p, n):
        ar = sum(phi[i] * w[t - 1 - i] for i in range(p))
        ma = sum(theta[j] * e[t - 1 - j] for j in range(q) if t - 1 - j >= 0)
        e[t] = w[t] - c - ar - ma
    return e[p:]


def _psi(phi, theta, h):
    """MA(infinity) weights of the ARMA part, psi_0 = 1."""
    psi = np.zeros(h)
    psi[0] = 1.0
    for j in range(1, h):
        psi[j] = (theta[j - 1] if j - 1 < len(theta) else 0.0) + sum(
            phi[i] * psi[j - 1 - i] for i in range(len(phi)) if j - 1 - i >= 0)
    return psi


@dataclass
class ARIMAResult:
    order: tuple
    drift: bool
    params: dict
    bse: dict
    sigma2: float
    loglik: float
    aic: float
    bic: float
    nobs: int
    resid: np.ndarray
    _y: np.ndarray = field(repr=False)

    def _unpack(self):
        p, _, q = self.order
        c = self.params.get("drift", 0.0)
        phi = np.array([self.params[f"ar.L{i}"] for i in range(1, p + 1)])
        theta = np.array([self.params[f"ma.L{j}"] for j in range(1, q + 1)])
        return c, phi, theta

    def forecast(self, steps=1, alpha=0.05):
        """Point forecasts of the level and (lower, upper) prediction intervals."""
        c, phi, theta = self._unpack()
        w = list(np.diff(self._y))
        e = list(self.resid)
        p, q = len(phi), len(theta)
        fw = []
        for _ in range(steps):
            ar = sum(phi[i] * w[-1 - i] for i in range(p))
            ma = sum(theta[j] * e[-1 - j] for j in range(q))
            nxt = c + ar + ma
            fw.append(nxt)
            w.append(nxt)
            e.append(0.0)
        mean = self._y[-1] + np.cumsum(fw)
        cum_psi = np.cumsum(_psi(phi, theta, steps))
        se = np.sqrt(self.sigma2 * np.cumsum(cum_psi ** 2))
        z = stats.norm.ppf(1 - alpha / 2)
        return mean, mean - z * se, mean + z * se

    def summary(self):
        lines = [f"ARIMA{self.order}{' with drift' if self.drift else ''}   n = {self.nobs}",
                 f"{'param':<8}{'coef':>12}{'std err':>12}{'z':>9}{'P>|z|':>8}"]
        for k, v in self.params.items():
            s = self.bse[k]
            z = v / s if s > 0 else np.nan
            lines.append(f"{k:<8}{v:>12.4f}{s:>12.4f}{z:>9.2f}{2 * stats.norm.sf(abs(z)):>8.3f}")
        lines.append(f"sigma2 = {self.sigma2:,.1f}   log-lik = {self.loglik:,.2f}   "
                     f"AIC = {self.aic:,.2f}   BIC = {self.bic:,.2f}")
        return "\n".join(lines)


def fit(y, order=(0, 1, 1), drift=False):
    """Fit ARIMA(p, 1, q) to levels ``y`` by CSS. Returns an ARIMAResult."""
    p, d, q = order
    if d != 1:
        raise ValueError("This implementation covers d = 1 (prices in levels).")
    y = np.asarray(y, float)
    w = np.diff(y)

    def unpack(x):
        i = 0
        c = x[i] if drift else 0.0
        i += int(drift)
        return c, x[i:i + p], x[i + p:i + p + q]

    def css(x):
        c, phi, theta = unpack(x)
        if q and np.any(np.abs(np.roots(np.r_[1, theta][::-1])) <= 1.0001):
            return 1e30                               # keep the MA part invertible
        e = _residuals(w, c, phi, theta)
        return e @ e

    x0 = np.r_[[w.mean()] if drift else [], np.zeros(p + q)]
    if len(x0):
        res = optimize.minimize(css, x0, method="Nelder-Mead",
                                options=dict(xatol=1e-8, fatol=1e-8, maxiter=20000))
        x = res.x
    else:
        x = np.zeros(0)
    c, phi, theta = unpack(x)
    e = _residuals(w, c, phi, theta)
    n = len(e)
    sigma2 = e @ e / n
    loglik = -0.5 * n * (np.log(2 * np.pi * sigma2) + 1)

    # standard errors from the numerical Hessian of the concentrated log-likelihood
    def nll(xx):
        s = css(xx)
        return 0.5 * n * np.log(s / n)
    bse = np.full(len(x), np.nan)
    if len(x):
        h = np.maximum(1e-4 * np.abs(x), 1e-5)
        H = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            for j in range(len(x)):
                ei, ej = np.eye(len(x))[i] * h[i], np.eye(len(x))[j] * h[j]
                H[i, j] = (nll(x + ei + ej) - nll(x + ei - ej) - nll(x - ei + ej) + nll(x - ei - ej)) / (4 * h[i] * h[j])
        try:
            bse = np.sqrt(np.diag(np.linalg.inv(H)))
        except np.linalg.LinAlgError:
            pass

    names = (["drift"] if drift else []) + [f"ar.L{i}" for i in range(1, p + 1)] + [f"ma.L{j}" for j in range(1, q + 1)]
    k = len(x) + 1                                    # + sigma2, as statsmodels counts it
    return ARIMAResult(order=order, drift=drift, params=dict(zip(names, x)), bse=dict(zip(names, bse)),
                       sigma2=float(sigma2), loglik=float(loglik), aic=float(2 * k - 2 * loglik),
                       bic=float(k * np.log(n) - 2 * loglik), nobs=n, resid=e, _y=y)


def select_order(y, max_p=2, max_q=2, drift_options=(False, True)):
    """
    Grid search over (p, 1, q) and drift by AIC (like pmdarima.auto_arima, but exhaustive).

    CSS conditions on the first p observations, so every model is fitted on the same evaluation
    sample (the first max_p differences are used only as initial values); otherwise models with
    more AR lags would be scored on fewer, possibly calmer, observations.
    """
    y = np.asarray(y, float)
    rows = []
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            for drift in drift_options:
                r = fit(y[max_p - p:], (p, 1, q), drift)
                rows.append(dict(order=(p, 1, q), drift=drift, aic=r.aic, bic=r.bic, nobs=r.nobs))
    rows.sort(key=lambda r: r["aic"])
    return rows
