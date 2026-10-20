"""Forecast evaluation: accuracy metrics, rolling one-step forecasts, Diebold-Mariano test."""
import numpy as np
import pandas as pd
from scipy import stats

from . import arima


def metrics(actual, forecast):
    a, f = np.asarray(actual, float), np.asarray(forecast, float)
    e = a - f
    return dict(RMSE=float(np.sqrt(np.mean(e ** 2))), MAE=float(np.mean(np.abs(e))),
                MAPE=float(np.mean(np.abs(e / a)) * 100),
                sMAPE=float(np.mean(2 * np.abs(e) / (np.abs(a) + np.abs(f))) * 100))


def rolling_one_step(y, n_train, order=(0, 1, 1), drift=False, refit=True):
    """
    Expanding-window one-step-ahead forecasts for y[n_train:].
    refit=True re-estimates the model every day on all data up to the previous close.
    Returns a DataFrame with actual, arima, naive (random walk) and the ARIMA interval.
    """
    y = pd.Series(y).astype(float)
    rows = []
    res = arima.fit(y.iloc[:n_train].values, order, drift)
    for t in range(n_train, len(y)):
        if refit and t > n_train:
            res = arima.fit(y.iloc[:t].values, order, drift)
        elif t > n_train:                            # keep parameters, update the residuals only
            res = arima.ARIMAResult(**{**res.__dict__, "_y": y.iloc[:t].values,
                                       "resid": arima._residuals(np.diff(y.iloc[:t].values),
                                                                 *res._unpack())})
        mean, lo, hi = res.forecast(1)
        rows.append(dict(date=y.index[t], actual=y.iloc[t], arima=mean[0], lower=lo[0], upper=hi[0],
                         naive=y.iloc[t - 1]))
    return pd.DataFrame(rows).set_index("date")


def diebold_mariano_losses(loss1, loss2, h=1):
    """
    Diebold-Mariano test on two series of forecast losses, with the Harvey-Leybourne-Newbold
    small-sample correction. Negative statistic: model 1 has lower loss. Returns (stat, two-sided p).
    """
    d = np.asarray(loss1, float) - np.asarray(loss2, float)
    n = len(d)
    dbar = d.mean()
    gamma = [np.mean((d[k:] - dbar) * (d[:n - k] - dbar)) for k in range(h)]
    var = (gamma[0] + 2 * sum(gamma[1:])) / n
    if var <= 0:                          # identical losses: no evidence either way
        return 0.0, 1.0
    dm = dbar / np.sqrt(var)
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    return float(dm), float(2 * stats.t.sf(abs(dm), n - 1))


def diebold_mariano(actual, f1, f2, h=1, power=2):
    """DM test for point forecasts with loss |error|^power (power=2: squared error)."""
    a = np.asarray(actual, float)
    return diebold_mariano_losses(np.abs(a - np.asarray(f1)) ** power, np.abs(a - np.asarray(f2)) ** power, h)


def direction_accuracy(actual, forecast, previous):
    """Share of days on which the forecast gets the sign of the next move right (ties excluded)."""
    a = np.sign(np.asarray(actual) - np.asarray(previous))
    f = np.sign(np.asarray(forecast) - np.asarray(previous))
    m = (a != 0) & (f != 0)
    return float(np.mean(a[m] == f[m])) if m.any() else float("nan")


def predictive_regression_oos(target, features, n_train):
    """
    Expanding-window OLS forecasts of ``target`` from ``features`` (both aligned so that features
    at row t are known before target at row t). Returns (in-sample t-stats on the training window,
    out-of-sample R^2 against a zero forecast, i.e. against the random walk for price changes).
    """
    y = np.asarray(target, float)
    X = np.column_stack([np.ones(len(y)), np.asarray(features, float)])
    preds = []
    for t in range(n_train, len(y)):
        b, *_ = np.linalg.lstsq(X[:t], y[:t], rcond=None)
        preds.append(X[t] @ b)
    preds = np.array(preds)
    act = y[n_train:]
    r2_oos = 1 - np.sum((act - preds) ** 2) / np.sum(act ** 2)
    b, *_ = np.linalg.lstsq(X[:n_train], y[:n_train], rcond=None)
    e = y[:n_train] - X[:n_train] @ b
    se = np.sqrt(np.diag(e @ e / (n_train - X.shape[1]) * np.linalg.inv(X[:n_train].T @ X[:n_train])))
    return b / se, float(r2_oos)
