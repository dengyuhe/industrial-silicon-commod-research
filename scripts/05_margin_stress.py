"""
Step 5: producer margins and stress tests.

Margin = selling price - full production cost. Shocks: electricity tariff, raw-material prices,
product price, one at a time and combined; a price x tariff grid; and a one-month margin
distribution from GARCH(1,1)-t simulated price paths.

Selling prices: 553# spot in Xinjiang for Xinjiang producers, in Kunming for Southwest producers.

Outputs: results/margins.csv, results/stress_tests.csv, results/margin_risk.csv,
         figures/margin_stress.png, figures/margin_risk.png
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import BLUE, END_DATE, GREY, INK, ORANGE, RED, RES, save
from sifutures import data as D, fundamentals as F, volatility as V

spot = D.load_spot_and_contracts()
xj, km = spot["spot_553_xinjiang"].dropna(), spot["spot_553_kunming"].dropna()
REFS = {"2024 average": (xj.loc["2024"].mean(), km.loc["2024"].mean()),
        "2025 low (4 Jun)": (xj.loc["2025-06-04"], km.loc["2025-06-04"]),
        "Latest (16 Jul 2025)": (xj.iloc[-1], km.iloc[-1])}
MARKET = {"Hoshine, Shanshan base (2024)": 0, "Xinjiang average": 0,
          "Southwest, wet season": 1, "Southwest, dry season": 1}

# --- Margins at reference prices ------------------------------------------------------------------------
rows = []
for sc, i in MARKET.items():
    for ref, prices in REFS.items():
        p = prices[i]
        m, pct = F.margin(sc, p)
        rows.append(dict(producer=sc, price_reference=ref, price=p, full_cost=F.total_cost(sc), margin=m,
                         margin_pct=100 * pct, breakeven_tariff=F.breakeven_tariff(sc, p),
                         actual_tariff=F.COST_SCENARIOS[sc]["Electricity"][1]))
margins = pd.DataFrame(rows)
margins.to_csv(RES / "margins.csv", index=False)
print(margins.round(3).to_string(index=False))

latest = {sc: REFS["Latest (16 Jul 2025)"][i] for sc, i in MARKET.items()}
stress = pd.concat([F.stress_scenarios(sc, p) for sc, p in latest.items()], axis=1)
stress.to_csv(RES / "stress_tests.csv")
print(stress.round(0).to_string())

# --- Figure: tornado + price x tariff grid ----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1, 1.1]})
ax = axes[0]
sc = "Xinjiang average"
base = stress.loc["Base", sc]
pairs = [("Product price ±10%", "Product price -10%", "Product price +10%"),
         ("Raw materials ±20%", "Raw materials +20%", "Raw materials -20%"),
         ("Electricity ±0.05 CNY/kWh", "Electricity +0.05 CNY/kWh", "Electricity -0.05 CNY/kWh")]
for k, (lab, bad, good) in enumerate(pairs):
    lo, hi = stress.loc[bad, sc], stress.loc[good, sc]
    ax.barh(k, lo - base, left=base, color=RED, alpha=0.75, height=0.55)
    ax.barh(k, hi - base, left=base, color=BLUE, alpha=0.75, height=0.55)
    ax.text(lo - 60, k, f"{lo:,.0f}", ha="right", va="center", fontsize=8)
    ax.text(hi + 60, k, f"{hi:,.0f}", ha="left", va="center", fontsize=8)
ax.axvline(base, color=INK, lw=1)
ax.axvline(0, color=GREY, lw=0.8, ls="--")
ax.set_yticks(range(len(pairs)), [p[0] for p in pairs])
ax.set(title=f"Xinjiang average producer: margin under shocks\n(base {base:,.0f} CNY/t at 8,650 CNY/t spot)",
       xlabel="Margin (CNY/t)", xlim=(-2200, 1000))
ax.grid(axis="y", visible=False)

ax = axes[1]
prices = np.arange(7000, 13001, 500)
tariffs = np.round(np.arange(0.20, 0.551, 0.05), 2)
grid = np.array([[p - F.shocked_cost(sc, elec_price=t) for p in prices] for t in tariffs])
lim = np.abs(grid).max()
im = ax.imshow(grid, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto", origin="lower",
               extent=[prices[0] - 250, prices[-1] + 250, tariffs[0] - 0.025, tariffs[-1] + 0.025])
ax.contour(prices, tariffs, grid, levels=[0], colors=INK, linewidths=1.5)
ax.plot(latest[sc], F.COST_SCENARIOS[sc]["Electricity"][1], "o", color="white", mec=INK, ms=7,
        label="Xinjiang average, 16 Jul 2025")
ax.set(title="Margin (CNY/t) by selling price and electricity tariff\n(Xinjiang average cost structure; black line = breakeven)",
       xlabel="Selling price (CNY/t)", ylabel="Electricity tariff (CNY/kWh)")
ax.grid(False)
ax.legend(frameon=True, fontsize=7.5, loc="upper left")
fig.colorbar(im, ax=ax, fraction=0.045, label="CNY/t")
save(fig, "margin_stress.png")

# --- One-month margin distribution from GARCH(1,1)-t price paths ---------------------------------------
fut = D.load_futures().loc[:END_DATE]
gaps = D.roll_gaps(fut["close"], spot)
r = 100 * np.diff(np.log(D.roll_adjust(fut["close"], gaps, "ratio").values))
fit = V.fit(r, "garch", "t")
cum = V.simulate_returns(fit, n_days=21, n_paths=20000, seed=0)
rows = []
fig, ax = plt.subplots(figsize=(9.5, 3.8))
for (sc, p), color in zip(latest.items(), [BLUE, INK, ORANGE, RED]):
    m = p * np.exp(cum / 100) - F.total_cost(sc)
    rows.append(dict(producer=sc, start_price=p, prob_loss=np.mean(m < 0), margin_p5=np.percentile(m, 5),
                     margin_median=np.median(m), margin_p95=np.percentile(m, 95)))
    ax.hist(m, bins=120, range=(-7000, 4000), histtype="step", lw=1.5, color=color,
            label=f"{sc}: P(loss) = {np.mean(m < 0):.0%}", density=True)
ax.axvline(0, color=GREY, ls="--", lw=1)
ax.set(title="Margin in one month (21 trading days), simulated from GARCH(1,1)-t price paths",
       xlabel="Margin (CNY/t)", ylabel="Density")
ax.legend(frameon=False, fontsize=7.5, loc="upper left")
save(fig, "margin_risk.png")
risk = pd.DataFrame(rows)
risk.attrs["note"] = "zero drift; futures volatility applied to spot prices"
risk.to_csv(RES / "margin_risk.csv", index=False)
print(risk.round(2).to_string(index=False))
print("GARCH-t full-sample:", {k: round(v, 3) for k, v in fit.params.items()},
      "| 21-day 5/95% cumulative return:", np.percentile(cum, [5, 95]).round(1))
