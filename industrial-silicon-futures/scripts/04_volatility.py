"""
Step 4: volatility models for daily log returns of the roll-adjusted series.

Compares constant variance, EWMA, GARCH(1,1) and GJR-GARCH(1,1) with normal or Student-t errors:
in-sample fit on the training period, then expanding-window one-day-ahead forecasts (re-estimated
daily) over the test period, judged by QLIKE loss and by the frequency of 95%-interval breaches.

Outputs: results/volatility_insample.csv, results/volatility_oos.csv, figures/volatility.png
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BLUE, END_DATE, GREY, INK, N_TRAIN, ORANGE, RED, RES, save
from sifutures import data as D, evaluate as ev, timeseries as ts, volatility as V

fut = D.load_futures().loc[:END_DATE]
gaps = D.roll_gaps(fut["close"], D.load_spot_and_contracts())
price = D.roll_adjust(fut["close"], gaps, "ratio")              # ratio adjustment preserves returns
r = pd.Series(100 * np.diff(np.log(price.values)), index=price.index[1:], name="return_pct")
n_tr = N_TRAIN - 1                                               # returns in the training period
train, test = r.iloc[:n_tr], r.iloc[n_tr:]

SPECS = {"Constant variance (normal)": ("constant", "normal"),
         "Constant variance (t)": ("constant", "t"),
         "EWMA (RiskMetrics, λ = 0.94)": ("ewma", "normal"),
         "GARCH(1,1) normal": ("garch", "normal"),
         "GARCH(1,1) t": ("garch", "t"),
         "GJR-GARCH(1,1) normal": ("gjr", "normal"),
         "GJR-GARCH(1,1) t": ("gjr", "t")}

# --- In-sample fit on the training period -------------------------------------------------------------
rows, fits = [], {}
for label, (model, dist) in SPECS.items():
    f = V.fit(train.values, model, dist)
    fits[label] = f
    z = f.std_resid
    rows.append({"model": label, "log-likelihood": f.loglik, "AIC": f.aic,
                 **{k: v for k, v in f.params.items() if k not in ("mu", "sigma2")},
                 "persistence": f.persistence if model != "constant" else np.nan,
                 "half-life (days)": f.half_life if model in ("garch", "gjr") else np.nan,
                 "ARCH-LM p on std. residuals": ts.arch_lm(z - z.mean(), 5)[1]})
ins = pd.DataFrame(rows).set_index("model")
ins["ΔAIC"] = ins["AIC"] - ins["AIC"].min()
ins.to_csv(RES / "volatility_insample.csv")
print(ins.round(3).to_string())

# --- Out-of-sample: daily re-estimation, one-day-ahead forecasts --------------------------------------
OOS = {k: SPECS[k] for k in ["Constant variance (normal)", "Constant variance (t)", "EWMA (RiskMetrics, λ = 0.94)",
                             "GARCH(1,1) normal", "GARCH(1,1) t", "GJR-GARCH(1,1) t"]}
fc = V.rolling_forecasts(r.values, n_tr, OOS, refit_every=1)
a = test.values
base = V.qlike(a ** 2, fc["Constant variance (normal)"]["var"])
rows = []
for label, v in fc.items():
    q = V.qlike(a ** 2, v["var"])
    hits = (a < v["lo"]) | (a > v["hi"])
    dm, p = ev.diebold_mariano_losses(q, base) if label != "Constant variance (normal)" else (np.nan, np.nan)
    rows.append({"model": label, "QLIKE": q.mean(), "QLIKE vs constant": q.mean() - base.mean(),
                 "DM p vs constant": p, "breaches (expected 6.25)": int(hits.sum()),
                 "coverage": 1 - hits.mean(), "Kupiec p": V.kupiec(hits, 0.05)[1],
                 "independence p": V.christoffersen_independence(hits)[1],
                 "avg interval width (%)": np.mean(v["hi"] - v["lo"])})
oos = pd.DataFrame(rows).set_index("model")
oos.to_csv(RES / "volatility_oos.csv")
print(oos.round(3).to_string())
print(f"daily sd of returns: training {train.std():.2f}%, test {test.std():.2f}%")

# --- Figure ---------------------------------------------------------------------------------------------
best = fits["GARCH(1,1) t"]
g = fc["GARCH(1,1) t"]
c = fc["Constant variance (normal)"]
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.1), gridspec_kw={"width_ratios": [1.35, 1]})
ax = axes[0]
ax.bar(r.index, r.abs(), width=1.5, color=GREY, alpha=0.6, label="|daily return|")
ax.plot(train.index, np.sqrt(best.sigma2), color=BLUE, lw=1.4, label="GARCH(1,1)-t volatility (in-sample)")
ax.plot(test.index, np.sqrt(g["var"]), color=ORANGE, lw=1.4, label="GARCH(1,1)-t forecast (out-of-sample)")
ax.axhline(train.std(), color=INK, ls="--", lw=0.9, label="Constant volatility (training)")
ax.set(title="Daily volatility of industrial silicon futures (%)", ylabel="%")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

ax = axes[1]
ax.plot(test.index, a, color=INK, lw=0.9, label="Daily return")
ax.fill_between(test.index, g["lo"], g["hi"], color=ORANGE, alpha=0.2, lw=0, label="GARCH(1,1)-t 95% interval")
ax.plot(test.index, c["lo"], color=BLUE, ls="--", lw=0.8, label="Constant-variance 95% interval")
ax.plot(test.index, c["hi"], color=BLUE, ls="--", lw=0.8)
hit = (a < g["lo"]) | (a > g["hi"])
ax.plot(test.index[hit], a[hit], "o", color=RED, ms=4, label=f"Breaches of GARCH interval ({hit.sum()})")
ax.set(title="Test period: interval forecasts for daily returns", ylabel="%")
ax.legend(frameon=False, fontsize=7.5, loc="lower left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
save(fig, "volatility.png")
