"""
Step 1: audit the active-contract series for contract-roll artifacts and build the adjusted series.

Outputs: results/roll_gaps.csv, results/futures_adjusted.csv, figures/roll_artifact.png
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np

from common import BLUE, END_DATE, GREY, INK, ORANGE, RES, save
from sifutures import data as D

fut = D.load_futures().loc[:END_DATE]
contracts = D.load_spot_and_contracts()
gaps = D.roll_gaps(fut["close"], contracts)
gaps.to_csv(RES / "roll_gaps.csv")
print(gaps.round(4).to_string())

out = fut.copy()
out["close_adj"] = D.roll_adjust(fut["close"], gaps, "difference")
out["close_adj_ratio"] = D.roll_adjust(fut["close"], gaps, "ratio")
out.to_csv(RES / "futures_adjusted.csv")

d = gaps.index[0]
win = contracts.loc[d - np.timedelta64(30, "D"): d + np.timedelta64(20, "D")]
fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.9))
ax = axes[0]
a = fut["close"].loc[win.index[0]:win.index[-1]]
ax.plot(a.index, a, color=BLUE, lw=5, alpha=0.35, label="Wind 'active contract' series")
ax.plot(win.index, win["nearby1_close"], color=INK, lw=1.1, label="Nearest contract")
ax.plot(win.index, win["nearby2_close"], color=ORANGE, lw=1.1, label="Second contract")
ax.axvline(d, color=INK, lw=0.8, ls="--")
ax.set(title=f"The active series switches contracts on {d:%d %b %Y}", ylabel="CNY/t")
ax.legend(frameon=False, fontsize=8)
ax.tick_params(axis="x", rotation=30)

ax = axes[1]
raw = fut["close"].diff()
adj = out["close_adj"].diff()
ax.plot(raw.index, raw, color=GREY, lw=0.8, label="Raw series")
ax.plot(adj.index, adj, color=BLUE, lw=0.8, label="Roll-adjusted")
ax.set(title="Daily price change: the +2,885 spike is the roll, not the market", ylabel="CNY/t")
ax.legend(frameon=False, fontsize=8)
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7)))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
save(fig, "roll_artifact.png")
