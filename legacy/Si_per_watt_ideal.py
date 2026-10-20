#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 14:47:10 2025

@author: dengyuhe
"""

import numpy as np
import matplotlib.pyplot as plt

# Parameters (varies as tech develops)
recycled_material = 28200    # t/a
ingot_output = 71424         # t/a
polysilicon_input = 104526   # t/a
density = 2.33e-3            # g/mm³
polish_thickness = 0.05      # mm

yield_wafer = 0.95
yield_battery = 0.98
yield_module = 0.98

silicon_price_increase = 1000  # yuan/ton

# list of wafer types
battery_types = [
    {
        "name": "M10-182",
        "a": 182, "b": 182, "c": 5.22, "d": 5.22,
        "thickness": 0.15, "efficiency": 0.235
    },
    {
        "name": "G12-210-PERC",
        "a": 210, "b": 210, "c": 1.41, "d": 1.41,
        "thickness": 0.15, "efficiency": 0.234
    },
    {
        "name": "183R",
        "a": 183.75, "b": 182.2, "c": 8.57, "d": 8.49,
        "thickness": 0.125, "efficiency": 0.255
    },
    {
        "name": "G12-210-TOPCon",
        "a": 210, "b": 210, "c": 1.41, "d": 1.41,
        "thickness": 0.125, "efficiency": 0.256
    },
    {
        "name": "HJT",
        "a": 182, "b": 191.6, "c": 1.32, "d": 1.38,
        "thickness": 0.12, "efficiency": 0.255
    },
    {
        "name": "210R",
        "a": 182, "b": 210, "c": 3.93, "d": 4.57,
        "thickness": 0.125, "efficiency": 0.2585
    }
]



def loss_rate():
    total_output = recycled_material + ingot_output
    loss = 1 - total_output / polysilicon_input
    return loss / (1 - recycled_material / polysilicon_input)

def sectional_area(a, b, c, d):
    return a * b - 2 * c * d  # mm²

def mass_per_wafer(area, thickness):
    total_thickness = thickness + polish_thickness
    volume = area * total_thickness  # mm³
    return volume * density  # g

def power_per_wafer(area, efficiency):
    return (area / 1e6) * efficiency * 1000  # W

def wafer_silicon_per_watt(area, thickness, efficiency, loss_rate_value):
    mass = mass_per_wafer(area, thickness) * yield_wafer
    power = power_per_wafer(area, efficiency)
    g_per_watt = mass / power
    return g_per_watt / (1 - loss_rate_value)

def main():
    print("Battery Type       |   efficiency     | Si (g/W) ")
    print("-" * 75)

    loss = loss_rate()

    for cell in battery_types:
        name = cell["name"]
        a, b, c, d = cell["a"], cell["b"], cell["c"], cell["d"]
        thickness = cell["thickness"]
        efficiency = cell["efficiency"]

        area = sectional_area(a, b, c, d)
        si_per_watt = wafer_silicon_per_watt(area, thickness, efficiency, loss)

        print(f"{name:<18} | {efficiency:>16.6f} | {si_per_watt:>18.6f}")

# Run
if __name__ == "__main__":
    main()
