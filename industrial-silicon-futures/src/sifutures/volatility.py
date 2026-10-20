"""
Conditional-volatility models for daily returns, estimated by maximum likelihood (numpy/scipy).

Models for returns r_t (in percent) with constant mean mu, e_t = r_t - mu, e_t = sigma_t z_t:
  constant : sigma_t^2 = sigma^2
  EWMA     : sigma_t^2 = lam sigma_{t-1}^2 + (1 - lam) e_{t-1}^2         (RiskMetrics, lam = 0.94)
  GARCH    : sigma_t^2 = omega + alpha e_{t-1}^2 + beta sigma_{t-1}^2
  GJR-GARCH: sigma_t^2 = omega + (alpha + gamma 1[e_{t-1} < 0]) e_{t-1}^2 + beta sigma_{t-1}^2
z_t is standard normal or standardized Student-t with nu degrees of freedom.
"""
from dataclasses import dataclass

import numpy as np
from scipy import optimize, signal, special, stats


def _filter(e, omega, alpha, gamma, beta, h0):
    """
    Variance recursion h[t+1] = omega + (alpha + gamma 1[e_t < 0]) e_t^2 + beta h[t], h[0] = h0.
    It is linear in h, so it runs as a single IIR filter. h[-1] is the next-day forecast.
    """
    x = omega + (alpha + gamma * (e < 0)) * e ** 2
    tail, _ = signal.lfilter([1.0], [1.0, -beta], x, zi=[beta * h0])
    return np.r_[h0, tail]


def _loglik(e, h, nu=None):
    if nu is None:
        return -0.5 * np.sum(np.log(2 * np.pi) + np.log(h) + e ** 2 / h)
    c = special.gammaln((nu + 1) / 2) - special.gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2))
    return np.sum(c - 0.5 * np.log(h) - (nu + 1) / 2 * np.log1p(e ** 2 / (h * (nu - 2))))


@dataclass
class VolResult:
    name: str
    params: dict
    loglik: float
    aic: float
    sigma2: np.ndarray        # in-sample conditional variances
    forecast_var: float       # next-day variance
    std_resid: np.ndarray
    nu: float = None

    @property
    def persistence(self):
        p = self.params
        if "lambda" in p:
            return 1.0                       # EWMA is an integrated (non-mean-reverting) model
        return p.get("alpha", 0) + p.get("gamma", 0) / 2 + p.get("beta", 0)

    @property
    def half_life(self):
        """Days for a volatility shock to decay by half."""
        ps = self.persistence
        return np.log(0.5) / np.log(ps) if 0 < ps < 1 else np.inf


def fit(r, model="garch", dist="normal", lam=0.94):
    """Fit one of 'constant', 'ewma', 'garch', 'gjr' with 'normal' or 't' errors."""
    r = np.asarray(r, float)
    mu = r.mean()
    e = r - mu
    v = e.var()
    use_t = dist == "t"

    if model == "constant":
        h = np.full(len(e) + 1, v)
        k, params, nu = 1, {"mu": mu, "sigma2": v}, None
        if use_t:
            res = optimize.minimize_scalar(lambda n: -_loglik(e, h[:-1], n), bounds=(2.05, 100), method="bounded")
            nu = res.x
            params["nu"] = nu
            k += 1
        ll = _loglik(e, h[:-1], nu)
        return _result(f"Constant variance ({dist})", params, ll, k + 1, h, e, nu)

    if model == "ewma":
        h = _filter(e, 0.0, 1 - lam, 0.0, lam, v)
        ll = _loglik(e, h[:-1])
        return _result("EWMA (λ = 0.94)", {"mu": mu, "lambda": lam}, ll, 1, h, e, None)

    asym = model == "gjr"

    def unpack(x):
        omega, alpha = np.exp(x[0]), x[1]
        gamma = x[2] if asym else 0.0
        beta = x[3] if asym else x[2]
        nu = 2.05 + np.exp(x[-1]) if use_t else None
        return omega, alpha, gamma, beta, nu

    def nll(x):
        omega, alpha, gamma, beta, nu = unpack(x)
        if alpha < 0 or beta < 0 or alpha + gamma < 0 or alpha + gamma / 2 + beta >= 0.9999:
            return 1e10
        h = _filter(e, omega, alpha, gamma, beta, v)
        return -_loglik(e, h[:-1], nu)

    x0 = [np.log(v * 0.05), 0.08] + ([0.04] if asym else []) + [0.85] + ([np.log(6.0)] if use_t else [])
    best = None
    for start in (x0, [np.log(v * 0.2), 0.15] + ([0.0] if asym else []) + [0.6] + ([np.log(10.0)] if use_t else [])):
        res = optimize.minimize(nll, start, method="Nelder-Mead",
                                options=dict(maxiter=8000, xatol=1e-7, fatol=1e-8))
        if best is None or res.fun < best.fun:
            best = res
    omega, alpha, gamma, beta, nu = unpack(best.x)
    h = _filter(e, omega, alpha, gamma, beta, v)
    params = {"mu": mu, "omega": omega, "alpha": alpha}
    if asym:
        params["gamma"] = gamma
    params["beta"] = beta
    if use_t:
        params["nu"] = nu
    name = ("GJR-GARCH(1,1)" if asym else "GARCH(1,1)") + (" t" if use_t else " normal")
    return _result(name, params, -best.fun, len(best.x) + 1, h, e, nu)


def _result(name, params, ll, k, h, e, nu):
    return VolResult(name=name, params=params, loglik=float(ll), aic=float(2 * k - 2 * ll),
                     sigma2=h[:-1], forecast_var=float(h[-1]), std_resid=e / np.sqrt(h[:-1]), nu=nu)


def quantile(nu, p):
    """Quantile of the error distribution scaled to unit variance."""
    if nu is None:
        return stats.norm.ppf(p)
    return stats.t.ppf(p, nu) * np.sqrt((nu - 2) / nu)


def rolling_forecasts(r, n_train, specs, refit_every=1):
    """
    Expanding-window one-day-ahead variance forecasts for r[n_train:].
    ``specs``: {label: (model, dist)}. Returns DataFrame-ready dict of arrays per label:
    variance forecast, lower and upper 95% return bounds, and the error distribution's nu.
    """
    r = np.asarray(r, float)
    out = {k: dict(var=[], lo=[], hi=[]) for k in specs}
    fits = {}
    for i, t in enumerate(range(n_train, len(r))):
        for label, (model, dist) in specs.items():
            if i % refit_every == 0 or label not in fits:
                fits[label] = fit(r[:t], model, dist)
                params = fits[label].params
            f = fits[label]
            if i % refit_every != 0:          # update the filter with fixed parameters
                f = _refilter(f, r[:t])
            mu, var = f.params["mu"], f.forecast_var
            out[label]["var"].append(var)
            out[label]["lo"].append(mu + quantile(f.nu, 0.025) * np.sqrt(var))
            out[label]["hi"].append(mu + quantile(f.nu, 0.975) * np.sqrt(var))
    return {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in out.items()}


def _refilter(f, r):
    p = f.params
    e = np.asarray(r) - p["mu"]
    if "lambda" in p:
        h = _filter(e, 0.0, 1 - p["lambda"], 0.0, p["lambda"], e.var())
    elif "omega" in p:
        h = _filter(e, p["omega"], p["alpha"], p.get("gamma", 0.0), p["beta"], e.var())
    else:
        h = np.full(len(e) + 1, p["sigma2"])
    return VolResult(f.name, p, f.loglik, f.aic, h[:-1], float(h[-1]), e / np.sqrt(h[:-1]), f.nu)


# --- Evaluation of variance forecasts ----------------------------------------------------------------
def qlike(r2, h):
    """QLIKE loss (robust to noise in squared returns as a volatility proxy). Lower is better."""
    return np.log(h) + r2 / h


def kupiec(hits, p):
    """Kupiec proportion-of-failures test: are interval breaches as frequent as expected? (LR, pvalue)"""
    n, x = len(hits), int(np.sum(hits))
    phat = x / n
    if x in (0, n):
        ll1 = 0.0
    else:
        ll1 = (n - x) * np.log(1 - phat) + x * np.log(phat)
    lr = -2 * ((n - x) * np.log(1 - p) + x * np.log(p) - ll1)
    return float(lr), float(stats.chi2.sf(lr, 1))


def christoffersen_independence(hits):
    """Are breaches clustered (one breach making the next more likely)? (LR, pvalue)"""
    h = np.asarray(hits, int)
    a, b = h[:-1], h[1:]
    n00, n01 = np.sum((a == 0) & (b == 0)), np.sum((a == 0) & (b == 1))
    n10, n11 = np.sum((a == 1) & (b == 0)), np.sum((a == 1) & (b == 1))
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(q, k0, k1):
        return (k0 * np.log(1 - q) if k0 and q < 1 else 0) + (k1 * np.log(q) if k1 and q > 0 else 0)
    lr = -2 * (ll(p, n00 + n10, n01 + n11) - ll(p01, n00, n01) - ll(p11, n10, n11))
    return float(lr), float(stats.chi2.sf(lr, 1))


def simulate_returns(f, n_days, n_paths=20000, seed=0, zero_mean=True):
    """
    Simulate cumulative log returns (%) over ``n_days`` from a fitted GARCH/GJR result, starting
    from its next-day variance forecast. Errors drawn from the model's (standardized) distribution.
    zero_mean=True drops the estimated drift, which reflects the sample's trend rather than a forecast.
    """
    p = f.params
    rng = np.random.default_rng(seed)
    mu = 0.0 if zero_mean else p["mu"]
    h = np.full(n_paths, f.forecast_var)
    cum = np.zeros(n_paths)
    for _ in range(n_days):
        if f.nu is None:
            z = rng.standard_normal(n_paths)
        else:
            z = rng.standard_t(f.nu, n_paths) * np.sqrt((f.nu - 2) / f.nu)
        e = np.sqrt(h) * z
        cum += mu + e
        h = p["omega"] + (p["alpha"] + p.get("gamma", 0.0) * (e < 0)) * e ** 2 + p["beta"] * h
    return cum
