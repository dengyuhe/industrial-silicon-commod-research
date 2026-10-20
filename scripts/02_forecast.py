"""
Step 2: stationarity and volatility diagnostics, ARIMA estimation, and out-of-sample evaluation
against a random-walk benchmark. Runs on the raw series (to replicate the report) and on the
roll-adjusted series.

Outputs: results/diagnostics.csv, results/forecast_metrics.csv, results/order_search.csv,
         results/arima_summary.txt, figures/*.png
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BLUE, END_DATE, GREY, INK, N_TRAIN, ORANGE, RED, RES, save
from sifutures import arima, data as D, evaluate as ev, timeseries as ts

fut = D.load_futures().loc[:END_DATE]
gaps = D.roll_gaps(fut["close"], D.load_spot_and_contracts())
series = {"raw": fut["close"], "adjusted": D.roll_adjust(fut["close"], gaps, "difference")}
log_ret = {"raw": np.diff(np.log(fut["close"].values)),
           "adjusted": np.diff(np.log(D.roll_adjust(fut["close"], gaps, "ratio").values))}

# --- Diagnostics -------------------------------------------------------------------------------------
diag = {}
for k, y in series.items():
    r = log_ret[k] - log_ret[k].mean()
    jb, jbp, skew, kurt = ts.jarque_bera(r)
    diag[k] = {"ADF p (price level)": ts.adf(y.values)["pvalue"],
               "ADF p (first difference)": ts.adf(np.diff(y.values))["pvalue"],
               "ARCH-LM p (log returns, 5 lags)": ts.arch_lm(r, 5)[1],
               "ARCH-LM p (log returns, 10 lags)": ts.arch_lm(r, 10)[1],
               "Skewness (log returns)": skew, "Kurtosis (log returns)": kurt}
diag = pd.DataFrame(diag)
diag.to_csv(RES / "diagnostics.csv")
print(diag.round(4).to_string())

y = series["adjusted"]
train, test = y.iloc[:N_TRAIN], y.iloc[N_TRAIN:]

# --- Model selection on the training sample ---------------------------------------------------------
search = pd.DataFrame(arima.select_order(train.values, 2, 2))
search["delta_aic"] = search["aic"] - search["aic"].min()
search.to_csv(RES / "order_search.csv", index=False)
print(search.head(8).round(2).to_string(index=False))

w = np.diff(train.values)
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
band = 1.96 / np.sqrt(len(w))
for ax, (name, vals) in zip(axes, [("ACF", ts.acf(w, 30)), ("PACF", ts.pacf(w, 30))]):
    ax.axhspan(-band, band, color=BLUE, alpha=0.12, lw=0)
    ax.vlines(range(1, 31), 0, vals[1:], color=INK, lw=1.2)
    ax.plot(range(1, 31), vals[1:], "o", ms=3, color=INK)
    ax.axhline(0, color=INK, lw=0.6)
    ax.set(title=f"{name} of daily price changes (training sample)", xlabel="Lag (trading days)", ylim=(-0.25, 0.25))
save(fig, "acf_pacf.png")

model = arima.fit(train.values, (0, 1, 1))
with open(RES / "arima_summary.txt", "w") as fh:
    fh.write(model.summary() + "\n")
    fh.write(f"Ljung-Box (lags 1, 5, 10): {ts.ljung_box(model.resid, (1, 5, 10), dof=1)}\n")
print(model.summary())

# --- Static multi-step forecast (as in the report) -----------------------------------------------------
mean, lo, hi = model.forecast(len(test))
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train.index[-120:], train.iloc[-120:], color=GREY, lw=1.2, label="Training data")
ax.plot(test.index, test, color=INK, lw=1.4, label="Actual (test)")
ax.plot(test.index, mean, color=BLUE, lw=1.6, label="ARIMA(0,1,1) forecast")
ax.fill_between(test.index, lo, hi, color=BLUE, alpha=0.12, lw=0, label="95% interval")
ax.set(title="Multi-step forecast from 10 Jan 2025: flat, because the model is close to a random walk",
       ylabel="CNY/t (roll-adjusted)")
ax.legend(frameon=False, fontsize=8, loc="lower left")
save(fig, "static_forecast.png")

# --- Rolling one-step forecasts vs the random walk ------------------------------------------------------
rows, rolls = [], {}
for name, order in [("ARIMA(0,1,1)", (0, 1, 1)), ("ARIMA(2,1,0)", (2, 1, 0))]:
    roll = ev.rolling_one_step(y, N_TRAIN, order, refit=True)
    rolls[name] = roll
    m = ev.metrics(roll.actual, roll.arima)
    dm, p = ev.diebold_mariano(roll.actual, roll.arima, roll.naive)
    rows.append({"model": name, **m, "RMSE vs random walk (%)": (m["RMSE"] / ev.metrics(roll.actual, roll.naive)["RMSE"] - 1) * 100,
                 "DM stat": dm, "DM p": p,
                 "direction accuracy": ev.direction_accuracy(roll.actual, roll.arima, roll.naive),
                 "95% interval coverage": float(np.mean((roll.actual >= roll.lower) & (roll.actual <= roll.upper)))})
naive = rolls["ARIMA(0,1,1)"]
rows.append({"model": "Random walk (tomorrow = today)", **ev.metrics(naive.actual, naive.naive)})
static = ev.metrics(test.values, mean)
rows.append({"model": "ARIMA(0,1,1) static multi-step", **static,
             "95% interval coverage": float(np.mean((test.values >= lo) & (test.values <= hi)))})

# ARMA on log returns (= ARIMA on log prices): is there predictability in percentage returns?
logp = np.log(D.roll_adjust(fut["close"], gaps, "ratio"))
for name, order in [("ARMA(0,1) on log returns", (0, 1, 1)), ("ARMA(1,0) on log returns", (1, 1, 0))]:
    rl = ev.rolling_one_step(logp, N_TRAIN, order, refit=True)
    ra, rp = rl.actual - rl.naive, rl.arima - rl.naive              # realized and predicted returns
    dm, p = ev.diebold_mariano(rl.actual, rl.arima, rl.naive)
    rows.append({"model": name, "OOS R2 vs random walk (%)": 100 * (1 - np.sum((ra - rp) ** 2) / np.sum(ra ** 2)),
                 "DM stat": dm, "DM p": p, "direction accuracy": ev.direction_accuracy(rl.actual, rl.arima, rl.naive)})

# open interest and volume: do yesterday's changes predict today's price change?
d = pd.DataFrame({"dp": y.diff(), "doi": np.log(fut["open_interest"]).diff().shift(),
                  "dvol": np.log(fut["volume"]).diff().shift()}).dropna()
tstats, r2 = ev.predictive_regression_oos(d["dp"], d[["doi", "dvol"]], N_TRAIN - 2)
rows.append({"model": "Regression on lagged open interest & volume", "OOS R2 vs random walk (%)": r2 * 100})
metrics = pd.DataFrame(rows).set_index("model")
metrics.to_csv(RES / "forecast_metrics.csv")
print(metrics.round(3).to_string())
json.dump({"oi_volume_train_tstats": dict(zip(["const", "d_log_oi", "d_log_volume"], np.round(tstats, 2).tolist()))},
          open(RES / "oi_volume_regression.json", "w"), indent=1)

roll = rolls["ARIMA(0,1,1)"]
fig, axes = plt.subplots(1, 2, figsize=(12, 3.9))
ax = axes[0]
ax.plot(roll.index, roll.actual, color=INK, lw=1.3, label="Actual")
ax.plot(roll.index, roll.arima, color=BLUE, lw=1, ls="--", label="ARIMA one-step forecast")
ax.plot(roll.index, roll.naive, color=ORANGE, lw=1, ls=":", label="Random walk (yesterday's close)")
ax.set(title="Rolling one-step forecasts: ARIMA and the random walk overlap", ylabel="CNY/t")
ax.legend(frameon=False, fontsize=8)
ax.tick_params(axis="x", rotation=30)
ax = axes[1]
for name, color in [("ARIMA(0,1,1)", BLUE), ("ARIMA(2,1,0)", RED)]:
    r = rolls[name]
    gain = np.cumsum((r.actual - r.naive) ** 2 - (r.actual - r.arima) ** 2)
    ax.plot(r.index, gain / 1e3, color=color, lw=1.4, label=name)
ax.axhline(0, color=INK, lw=0.8)
ax.set(title="Cumulative squared-error advantage over the random walk", ylabel="thousand (CNY/t)²")
ax.legend(frameon=False, fontsize=8)
ax.tick_params(axis="x", rotation=30)
save(fig, "rolling_vs_random_walk.png")
