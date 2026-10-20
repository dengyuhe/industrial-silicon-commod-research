#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 14 09:22:17 2025

@author: dengyuhe
"""

unit_energy_range = (11000, 12500)      # kWh/t
unit_quartz_range = (2.6, 2.8)          # t/t         
unit_electrode_range = (0.08, 0.1)
wood_chip

REGIONAL_COSTS = {
    "Yunnan": {
        "electricity": (0.28, 0.38),      
        "quartz": (240, 270),      
        "carbon": (580, 620),          
        "labor": (180, 220),           
        "logistics": (160, 200),       
        "maintenance": (110, 130),       
    },
    "Xinjiang": {
        "electricity": (0.20, 0.30),
        "quartz": (220, 250),
        "carbon": (560, 600),
        "labor": (160, 200),
        "logistics": (230, 270),
        "maintenance": (100, 120),
    },
    "Sichuan": {
        "electricity": (0.30, 0.42),
        "quartz": (250, 280),
        "carbon": (600, 640),
        "labor": (200, 230),
        "logistics": (170, 210),
        "maintenance": (105, 125),
    }
}

# --- Main program ---
if __name__ == "__main__":
    print("=== Metal Silicon Production Cost Range Estimator ===")
    print("Available regions: Yunnan, Xinjiang, Sichuan")
    region = input("Enter a region name: ").strip()

    if region in REGIONAL_COSTS:
        model = SiliconCostModel(region, REGIONAL_COSTS[region])
        min_cost, max_cost = model.estimate_cost_range()
        print(f"\nEstimated cost range in {region}: ¥{min_cost} – ¥{max_cost} per ton")
    else:
        print(f"Region '{region}' not found.")