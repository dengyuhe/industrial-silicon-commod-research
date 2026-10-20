"""
Bottom-up fundamentals for Chinese industrial silicon (from the report, Chapters 2-3):
production cost by region, electricity sensitivity, cost-capacity curves, and silicon intensity
of solar modules. Costs in CNY per tonne of silicon, tax included.
"""
import numpy as np
import pandas as pd

# --- Production cost: unit consumption x unit price ---------------------------------------------
# (consumption per tonne of Si, unit price in CNY) or a fixed CNY amount
COST_SCENARIOS = {
    "Hoshine, Shanshan base (2024)": {
        "Silica": (2.7, 400), "Coke (washed)": (1.65, 900), "Wood chips": (0.4, 500),
        "Graphite electrode": (0.09, 9500), "Electricity": (11000, 0.28),
        "Labor": 400, "Other expenses": 1000, "Environmental fees": 200},
    "Xinjiang average": {
        "Silica": (2.7, 400), "Coke (washed)": (1.65, 900), "Wood chips": (0.6, 500),
        "Graphite electrode": (0.10, 9500), "Electricity": (11500, 0.32),
        "Labor": 400, "Other expenses": 1000, "Environmental fees": 200, "Sales/admin/R&D": 150},
    "Southwest, wet season": {
        "Silica": (2.5, 550), "Coke": (0.7, 1800), "Petroleum coke": (0.8, 1300), "Wood chips": (0.5, 450),
        "Graphite electrode": (0.08, 12000), "Electricity": (12000, 0.35),
        "Labor": 400, "Other expenses": 800, "Environmental fees": 200},
    "Southwest, dry season": {
        "Silica": (2.5, 550), "Coke": (0.7, 1800), "Petroleum coke": (0.8, 1300), "Wood chips": (0.5, 450),
        "Graphite electrode": (0.08, 12000), "Electricity": (12000, 0.53),
        "Labor": 400, "Other expenses": 800, "Environmental fees": 200},
}


def cost_breakdown(scenario):
    """Cost per component (CNY/t) and share of total for one scenario."""
    items = COST_SCENARIOS[scenario] if isinstance(scenario, str) else scenario
    cost = {k: (v[0] * v[1] if isinstance(v, tuple) else float(v)) for k, v in items.items()}
    s = pd.Series(cost, name="cost")
    return pd.DataFrame({"cost": s, "share": s / s.sum()})


def total_cost(scenario):
    return float(cost_breakdown(scenario)["cost"].sum())


def electricity_sensitivity(scenario="Hoshine, Shanshan base (2024)",
                            prices=np.round(np.arange(0.25, 0.3501, 0.01), 2),
                            usages=np.arange(10000, 13001, 500)):
    """Total cost (CNY/t) on a grid of electricity price (CNY/kWh) x usage (kWh/t)."""
    base = total_cost(scenario)
    kwh, price = COST_SCENARIOS[scenario]["Electricity"]
    other = base - kwh * price
    return pd.DataFrame([[other + p * u for u in usages] for p in prices], index=prices, columns=usages)


def electricity_elasticity(scenario="Hoshine, Shanshan base (2024)"):
    """
    Elasticity of total cost to the electricity price at fixed usage. It equals electricity's
    share of total cost: d ln C / d ln p = p*q / C.
    """
    b = cost_breakdown(scenario)
    return float(b.loc["Electricity", "share"])


# --- Cost-capacity curves --------------------------------------------------------------------------
TOP5_PRODUCERS = pd.DataFrame({
    "producer": ["Hoshine (Xinjiang)", "Tongwei (Yongxiang)", "GCL Technology", "East Hope", "Daqo New Energy"],
    "capacity_kt": [1220, 420, 480, 145, 305],
    "cost": [8300, 9500, 10000, 10500, 12500],
})


def cost_curve(df, capacity="capacity_kt", cost="cost"):
    """Sort by cost and add cumulative capacity (the supply curve's step edges)."""
    d = df.sort_values(cost).reset_index(drop=True)
    d["cum_capacity"] = d[capacity].cumsum()
    d["start"] = d["cum_capacity"] - d[capacity]
    return d


# Monte Carlo of a stylised industry: regional input price ranges (CNY) and recipes
_REGION_PRICES = {
    "Xinjiang": dict(electricity=(0.20, 0.28), coal=(400, 480), petcoke=(750, 800), charcoal=(1000, 1200), quartz=(420, 460)),
    "Yunnan": dict(electricity=(0.30, 0.38), coal=(600, 700), petcoke=(750, 800), charcoal=(1000, 1200), quartz=(440, 480)),
    "Sichuan": dict(electricity=(0.32, 0.42), coal=(650, 750), petcoke=(750, 800), charcoal=(900, 1050), quartz=(450, 490)),
}
_RECIPES = {"coal_only": dict(coal=(1.8, 2.2)),
            "coal_petcoke": dict(coal=(1.0, 1.2), petcoke=(0.5, 0.6)),
            "charcoal_petcoke": dict(charcoal=(1.2, 1.5), petcoke=(0.2, 0.3))}
_REGION_RECIPES = {"Xinjiang": ["coal_only"], "Sichuan": ["charcoal_petcoke", "coal_petcoke"],
                   "Yunnan": ["coal_petcoke", "coal_only"]}


def simulate_plants(n=50, seed=0):
    """Simulated plants with regional input prices, reductant recipes and capacities (t/yr)."""
    rng = np.random.default_rng(seed)
    u = lambda lo_hi: rng.uniform(*lo_hi)
    rows = []
    for i in range(n):
        region = rng.choice(list(_REGION_PRICES))
        recipe = rng.choice(_REGION_RECIPES[region])
        p = {k: u(v) for k, v in _REGION_PRICES[region].items()}
        cost = (p["electricity"] * rng.integers(10500, 12501) + p["quartz"] * u((2.5, 2.8))
                + sum(p[m] * u(q) for m, q in _RECIPES[recipe].items())
                + u((350, 450)) + u((1250, 1350)))            # labor, logistics
        rows.append(dict(plant=i, region=region, recipe=recipe,
                         capacity_kt=rng.integers(20000, 80001) / 1000, cost=cost))
    return pd.DataFrame(rows)


# --- Silicon intensity of solar modules --------------------------------------------------------------
# Wafer formats: side lengths a x b (mm), chamfer c x d (mm), thickness (mm), cell efficiency
CELL_TYPES = pd.DataFrame([
    ("M10-182 PERC", 182, 182, 5.22, 5.22, 0.150, 0.235),
    ("G12-210 PERC", 210, 210, 1.41, 1.41, 0.150, 0.234),
    ("183R", 183.75, 182.2, 8.57, 8.49, 0.125, 0.255),
    ("G12-210 TOPCon", 210, 210, 1.41, 1.41, 0.125, 0.256),
    ("HJT", 182, 191.6, 1.32, 1.38, 0.120, 0.255),
    ("210R", 182, 210, 3.93, 4.57, 0.125, 0.2585),
], columns=["cell", "a", "b", "c", "d", "thickness", "efficiency"])

SILICON_DENSITY = 2.33e-3          # g/mm^3
KERF_THICKNESS = 0.05              # mm lost to polishing/sawing per wafer
YIELD_WAFER, YIELD_CELL, YIELD_MODULE = 0.95, 0.98, 0.98
# Ingot-pulling material balance (t/yr) used to derive the polysilicon loss rate
POLY_INPUT, INGOT_OUTPUT, RECYCLED = 104_526, 71_424, 28_200


def polysilicon_loss_rate():
    loss = 1 - (RECYCLED + INGOT_OUTPUT) / POLY_INPUT
    return loss / (1 - RECYCLED / POLY_INPUT)


def silicon_intensity(cells=CELL_TYPES):
    """Polysilicon (g) per watt of wafer output, and the cost added per watt by +1,000 CNY/t silicon."""
    d = cells.copy()
    area = d["a"] * d["b"] - 2 * d["c"] * d["d"]                                 # mm^2
    mass = area * (d["thickness"] + KERF_THICKNESS) * SILICON_DENSITY * YIELD_WAFER   # g per wafer
    watts = area / 1e6 * d["efficiency"] * 1000                                   # W per wafer
    d["g_per_watt"] = mass / watts / (1 - polysilicon_loss_rate())
    kg_w = d["g_per_watt"] / 1000
    d["module_cost_per_watt_per_1000cny_t"] = kg_w / YIELD_CELL / YIELD_MODULE * (1000 / 1000)
    return d[["cell", "thickness", "efficiency", "g_per_watt", "module_cost_per_watt_per_1000cny_t"]]


# --- Margin stress tests ---------------------------------------------------------------------------------
RAW_MATERIALS = ["Silica", "Coke (washed)", "Coke", "Petroleum coke", "Wood chips", "Graphite electrode"]


def shocked_cost(scenario, elec_price=None, elec_delta=0.0, raw_mult=1.0, elec_usage=None):
    """
    Full cost (CNY/t) after shocks: set or shift the electricity tariff (CNY/kWh), change unit
    electricity consumption (kWh/t), and scale all raw-material costs by ``raw_mult``.
    """
    items = dict(COST_SCENARIOS[scenario])
    kwh, tariff = items["Electricity"]
    items["Electricity"] = (elec_usage or kwh, (elec_price if elec_price is not None else tariff) + elec_delta)
    cost = cost_breakdown(items)["cost"]
    cost.loc[cost.index.isin(RAW_MATERIALS)] *= raw_mult
    return float(cost.sum())


def margin(scenario, price, **shocks):
    """Margin per tonne (CNY/t) and as a share of price."""
    m = price - shocked_cost(scenario, **shocks)
    return m, m / price


def breakeven_tariff(scenario, price, raw_mult=1.0):
    """Electricity tariff (CNY/kWh) at which the margin is zero for a given selling price."""
    kwh, _ = COST_SCENARIOS[scenario]["Electricity"]
    non_elec = shocked_cost(scenario, elec_price=0.0, raw_mult=raw_mult)
    return (price - non_elec) / kwh


def stress_scenarios(scenario, price):
    """Margin (CNY/t) under one-at-a-time and combined shocks."""
    cases = {
        "Base": {},
        "Electricity +0.05 CNY/kWh": dict(elec_delta=0.05),
        "Electricity -0.05 CNY/kWh": dict(elec_delta=-0.05),
        "Raw materials +20%": dict(raw_mult=1.2),
        "Raw materials -20%": dict(raw_mult=0.8),
    }
    rows = {k: price - shocked_cost(scenario, **v) for k, v in cases.items()}
    rows["Product price -10%"] = 0.9 * price - shocked_cost(scenario)
    rows["Product price +10%"] = 1.1 * price - shocked_cost(scenario)
    rows["Combined adverse (all three)"] = 0.9 * price - shocked_cost(scenario, elec_delta=0.05, raw_mult=1.2)
    rows["Combined favourable (all three)"] = 1.1 * price - shocked_cost(scenario, elec_delta=-0.05, raw_mult=0.8)
    return pd.Series(rows, name=scenario)
