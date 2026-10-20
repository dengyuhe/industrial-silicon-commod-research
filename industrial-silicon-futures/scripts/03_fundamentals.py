"""
Step 3: fundamentals - production cost by region, electricity sensitivity, cost curve against
spot prices, the monthly supply-demand balance, and silicon intensity of solar modules.

Outputs: results/cost_breakdown.csv, results/monthly_balance.csv, results/silicon_intensity.csv,
         figures/*.png
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BLUE, GREY, INK, ORANGE, RED, RES, save
from sifutures import data as D, fundamentals as F

# --- Cost stacks by scenario -------------------------------------------------------------------------
GROUPS = {"Electricity": ["Electricity"],
          "Raw materials": ["Silica", "Coke (washed)", "Coke", "Petroleum coke", "Wood chips", "Graphite electrode"],
          "Labor": ["Labor"],
          "Other & environmental": ["Other expenses", "Environmental fees", "Sales/admin/R&D"]}
COLORS = {"Electricity": BLUE, "Raw materials": ORANGE, "Labor": GREY, "Other & environmental": "#c9ccd3"}
rows = []
for s in F.COST_SCENARIOS:
    b = F.cost_breakdown(s)["cost"]
    rows.append({g: b.reindex(items).sum() for g, items in GROUPS.items()} | {"scenario": s})
stack = pd.DataFrame(rows).set_index("scenario")
pd.concat({s: F.cost_breakdown(s) for s in F.COST_SCENARIOS}).to_csv(RES / "cost_breakdown.csv")

fig, ax = plt.subplots(figsize=(9, 3.3))
left = np.zeros(len(stack))
for g in GROUPS:
    ax.barh(stack.index, stack[g], left=left, color=COLORS[g], label=g, height=0.6)
    left += stack[g].values
for i, total in enumerate(left):
    elec = stack["Electricity"].iloc[i] / total
    ax.text(total + 120, i, f"{total:,.0f}  (electricity {elec:.0%})", va="center", fontsize=8.5)
ax.invert_yaxis()
ax.set(title="Full production cost per tonne by region (CNY/t, tax included)", xlim=(0, 15500))
ax.legend(frameon=False, fontsize=8, ncol=4, loc="lower right", bbox_to_anchor=(1, -0.32))
ax.grid(axis="y", visible=False)
save(fig, "cost_by_region.png")

# --- Electricity sensitivity heatmap ----------------------------------------------------------------
m = F.electricity_sensitivity()
fig, ax = plt.subplots(figsize=(8, 4.6))
im = ax.imshow(m.values, cmap="YlGnBu", aspect="auto", origin="upper")
for i in range(m.shape[0]):
    for j in range(m.shape[1]):
        v = m.values[i, j]
        ax.text(j, i, f"{v:,.0f}", ha="center", va="center", fontsize=7.5,
                color="white" if v > m.values.mean() + 0.3 * m.values.std() else INK)
ax.set_xticks(range(m.shape[1]), [f"{u:,}" for u in m.columns])
ax.set_yticks(range(m.shape[0]), [f"{p:.2f}" for p in m.index])
ax.set(xlabel="Electricity usage (kWh/t)", ylabel="Electricity price (CNY/kWh)",
       title="Total cost (CNY/t) around the Hoshine baseline of 8,300 (0.28 CNY/kWh, 11,000 kWh/t)")
ax.grid(False)
fig.colorbar(im, ax=ax, fraction=0.04, label="CNY/t")
save(fig, "electricity_sensitivity.png")

# --- Spot prices against regional cost levels --------------------------------------------------------
spot = D.load_spot_and_contracts()[["spot_553_xinjiang", "spot_553_kunming"]].dropna(how="all")
fig, axes = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [1.45, 1]})
ax = axes[0]
ax.plot(spot.index, spot["spot_553_kunming"], color=INK, lw=1.3, label="553# spot, Kunming (Yunnan)")
ax.plot(spot.index, spot["spot_553_xinjiang"], color=BLUE, lw=1.3, label="553# spot, Xinjiang")
for (name, c), ls in zip([(s, F.total_cost(s)) for s in F.COST_SCENARIOS], ["-", "--", "-.", ":"]):
    ax.axhline(c, color=RED, lw=0.9, ls=ls, alpha=0.8)
    ax.text(spot.index[0], c + 120, f"{name}: {c:,.0f}", fontsize=7.5, color=RED)
ax.set(title="Spot prices fell through most of the cost curve in 2025", ylabel="CNY/t")
ax.legend(frameon=False, fontsize=8, loc="upper right")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.tick_params(axis="x", rotation=30)

ax = axes[1]
cc = F.cost_curve(F.TOP5_PRODUCERS)
for _, r in cc.iterrows():
    ax.bar(r["start"], r["cost"], width=r["capacity_kt"], align="edge", color=BLUE, alpha=0.25, edgecolor=BLUE)
    ax.text(r["start"] + r["capacity_kt"] / 2, r["cost"] + 150, r["producer"].split(" (")[0], ha="center", fontsize=7.5)
ax.axhline(spot["spot_553_xinjiang"].dropna().min(), color=INK, ls="--", lw=0.9)
ax.text(30, spot["spot_553_xinjiang"].dropna().min() - 650, "2025 low of Xinjiang spot", fontsize=7.5)
ax.set(title="Cost curve of the five largest producers", xlabel="Cumulative capacity (kt/year)",
       ylabel="Estimated cost (CNY/t)", ylim=(0, 14000))
save(fig, "spot_vs_cost.png")

# --- Monthly supply-demand balance -------------------------------------------------------------------
bal = D.monthly_balance(D.load_monthly_indicators(), D.load_trade_and_inventory())
bal["spot_553_kunming"] = spot["spot_553_kunming"].resample("ME").mean().reindex(bal.index)
bal.to_csv(RES / "monthly_balance.csv")
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.bar(bal.index, bal["surplus"], width=20, color=np.where(bal["surplus"] > 0, ORANGE, BLUE), alpha=0.8)
ax.set(ylabel="Supply − demand (10,000 t)",
       title="Monthly balance: persistent oversupply while prices fell (demand from downstream output)")
ax2 = ax.twinx()
ax2.plot(bal.index, bal["spot_553_kunming"], color=INK, lw=1.5)
ax2.set_ylabel("553# spot, Kunming (CNY/t)")
ax2.grid(False)
ax2.spines["right"].set_visible(True)
save(fig, "monthly_balance.png")

# --- Silicon intensity of PV modules ------------------------------------------------------------------
si = F.silicon_intensity()
si.to_csv(RES / "silicon_intensity.csv", index=False)
print(si.round(4).to_string(index=False))
print(bal[["supply", "demand", "surplus"]].describe().round(2))

# --- Downstream output (replaces the static chart in the original report) -----------------------------
ind = D.load_monthly_indicators().resample("ME").last()
fig, ax = plt.subplots(figsize=(10, 3.6))
for col, label, color in [("primary_al_alloy_output", "Primary aluminum alloy", GREY),
                          ("dmc_output", "Organosilicon (DMC)", ORANGE),
                          ("polysilicon_output", "Polysilicon", BLUE)]:
    x = ind[col].dropna()
    ax.plot(x.index, x, color=color, lw=1.8, label=label)
ax.set(title="Monthly output of the three main downstream sectors", ylabel="10,000 t")
ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
save(fig, "downstream_output.png")
