#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 12:46:13 2025

@author: dengyuhe
"""

import random
import matplotlib.pyplot as plt

def prices(region):
    if region == "Xinjiang":
        return {
            "electricity": random.uniform(0.20, 0.28),
            "coal": random.uniform(400, 480),
            "petcoke": random.uniform(750, 800),
            "charcoal": random.uniform(1000, 1200),
            "quartz": random.uniform(420, 460),
        }
    elif region == "Yunnan":
        return {
            "electricity": random.uniform(0.30, 0.38),
            "coal": random.uniform(600, 700),
            "petcoke": random.uniform(750, 800),
            "charcoal": random.uniform(1000, 1200),
            "quartz": random.uniform(440, 480),
        }
    elif region == "Sichuan":
        return {
            "electricity": random.uniform(0.32, 0.42),
            "coal": random.uniform(650, 750),
            "petcoke": random.uniform(750, 800),
            "charcoal": random.uniform(900, 1050),
            "quartz": random.uniform(450, 490),
        }

def estimate_cost(region, method, material_prices):
    electricity_kwh = random.randint(10500, 12500)
    electricity_cost = material_prices["electricity"] * electricity_kwh

    quartz_ton = random.uniform(2.5, 2.8)
    quartz_cost = material_prices["quartz"] * quartz_ton  

    method_usage = {
        'coal_only': {'coal': random.uniform(1.8, 2.2)},
        'coal_petcoke': {'coal': random.uniform(1.0, 1.2),
                         'petcoke': random.uniform(0.5, 0.6)},
        'charcoal_petcoke': {'charcoal': random.uniform(1.2, 1.5),
                             'petcoke': random.uniform(0.2, 0.3)},
    }

    carbon_cost = sum(material_prices[mat] * usage for mat, usage in method_usage[method].items())

    labor = random.uniform(350, 450)
    logistics = random.uniform(1250, 1350)

    return electricity_cost + quartz_cost + carbon_cost + labor + logistics

# Simulate N factories
N = 50
factories = []

for i in range(N):
    region = random.choice(["Yunnan", "Xinjiang", "Sichuan"])

    if region == "Xinjiang":
        method = "coal_only"
    elif region == "Sichuan":
        method = random.choice(["charcoal_petcoke", "coal_petcoke"])
    else:
        method = random.choice(["coal_petcoke", "coal_only"])

    capacity = random.randint(20000, 80000)

    material_prices = prices(region)  # Call the function here

    cost_per_ton = estimate_cost(region, method, material_prices)

    factories.append({
        "id": i,
        "region": region,
        "method": method,
        "capacity": capacity,
        "cost_per_ton": cost_per_ton
    })

# Sort by cost
sorted_factories = sorted(factories, key=lambda x: x["cost_per_ton"])

# Accumulate capacity
cumulative_capacity = 0
x_vals = []
y_vals = []

for factory in sorted_factories:
    cumulative_capacity += factory["capacity"]
    x_vals.append(cumulative_capacity / 1e6)  # millions of tons
    y_vals.append(factory["cost_per_ton"])

# Plot
plt.figure(figsize=(10, 6))
plt.step(x_vals, y_vals, where='post', linewidth=2)
plt.xlabel("Cumulative Capacity (Million Tons)")
plt.ylabel("Cost per Ton (Yuan)")
plt.title("Cost-Capacity Curve of Metal Silicon Production")
plt.grid(True)
plt.tight_layout()
plt.show()




import pandas as pd
import matplotlib.pyplot as plt

# Producer data
data = {
    "Producer": [
        "Hoshine (Xinjiang)", 
        "Tongwei (Yongxiang)", 
        "GCL Technology", 
        "East Hope", 
        "Daqo New Energy"
    ],
    "Capacity": [1220, 420, 480, 145, 305],  # in kt/year
    "Cost": [8300, 9500, 10000, 10500, 12500]
}

df = pd.DataFrame(data)
df = df.sort_values("Cost")
df["CumCap"] = df["Capacity"].cumsum()

# Generate step curve
x = []
y = []
for i, row in df.iterrows():
    prev = df["CumCap"].shift(fill_value=0).loc[i]
    x += [prev, row["CumCap"]]
    y += [row["Cost"], row["Cost"]]

plt.figure(figsize=(10, 5))
plt.step(x, y, where="post", linewidth=2, color='green')
plt.fill_between(x, y, step="post", alpha=0.2, color='green')

# Label
for idx, row in df.iterrows():
    mid = row["CumCap"] - row["Capacity"]/2
    plt.text(mid, row["Cost"] + 200, row["Producer"], ha="center")

plt.xlabel("Cumulative Capacity (kt/year)")
plt.ylabel("Estimated Cost (Yuan/t)")
plt.title("Cost–Capacity Curve: Top 5 Chinese Industrial Silicon Producers")
plt.grid(linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


