# Original research snippets

Scripts written during the original research (July 2025), kept for provenance. Each has been
rebuilt inside the `sifutures` package; the numbers they print are reproduced by the tests.

| Original script | Now lives in | Notes |
|---|---|---|
| `electricity_price_sensitivity.py` | `fundamentals.electricity_sensitivity` | Same grid and baseline (8,300 CNY/t) |
| `simulate_cost_capacity.py` | `fundamentals.simulate_plants`, `fundamentals.cost_curve`, `TOP5_PRODUCERS` | Seeded random generator added; the simulation omits electrodes and overheads, so its costs are a lower bound |
| `Si_per_watt_ideal.py`, `cost_transmission.py` | `fundamentals.silicon_intensity` | The two scripts shared most of their code; merged |
| `metal_silicon_cost.py` | `fundamentals.COST_SCENARIOS`, `cost_breakdown` | Unfinished draft (referenced an undefined class); replaced by the report's cost tables |

The ARIMA forecasting code from the original study was not preserved; `scripts/02_forecast.py`
rebuilds it from the report and reproduces its test-set results (RMSE 1,867, MAE 1,498, sMAPE 16.03%).
