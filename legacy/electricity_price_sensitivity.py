#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 31 15:44:06 2025

@author: dengyuhe
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Parameters
base_price = 0.28
base_usage = 11000
base_cost = 8300
base_elec_cost = base_price * base_usage

# Ranges
electricity_prices = np.round(np.arange(0.25, 0.36, 0.01), 2)
electricity_usages = np.arange(10000, 13001, 500)

# Matrix
sensitivity_matrix = pd.DataFrame(index=electricity_prices, columns=electricity_usages)

for p in electricity_prices:
    for u in electricity_usages:
        cost = base_cost - base_elec_cost + p * u
        sensitivity_matrix.loc[p, u] = round(cost, 0)

sensitivity_matrix = sensitivity_matrix.astype(float)

# Plot
plt.figure(figsize=(10, 6))
sns.heatmap(sensitivity_matrix, annot=True, fmt=".0f", cmap="YlGnBu", cbar_kws={'label': 'Total Cost (Yuan/t)'})
plt.title("Electricity Cost Sensitivity Matrix (Total Cost in Yuan/t)")
plt.xlabel("Electricity Usage (kWh/t)")
plt.ylabel("Electricity Price (Yuan/kWh)")
plt.tight_layout()
plt.savefig("electricity_sensitivity_heatmap.png", dpi=300)
plt.show()
