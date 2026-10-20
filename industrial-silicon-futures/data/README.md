# Data

The raw data are licensed vendor data (Wind and SMM) and are **not included** in this repository.
To reproduce the analysis, place the files in `data/raw/` under the names below. The loaders in
`src/sifutures/data.py` translate every Chinese header into the English names listed here.

## Files

| File in `data/raw/` | Source | Frequency | Content |
|---|---|---|---|
| `期货收盘价_活跃合约__工业硅.csv` | Wind | Daily, 22 Dec 2022 – 1 Aug 2025 | Industrial silicon futures, "active contract" series: close, open interest, volume |
| `Si553_price.xlsx` | SMM / Wind | Daily | 553# spot prices by region; nearest and second-nearest futures contracts |
| `ouput_monthly.xlsx` | SMM | Monthly and weekly | Output, operating rates, downstream production |
| `other_metrics.xlsx` | SMM | Daily to monthly | PV chain prices, social inventory, exports and imports |

Industrial silicon futures trade on the **Guangzhou Futures Exchange (GFEX)**, which listed them on
22 December 2022. Daily settlement data for individual contracts are published by the exchange,
which is the natural public substitute for the Wind series.

## Data dictionary (Chinese source name → English column)

**Futures and spot**

| Source name | Column | Unit |
|---|---|---|
| 收盘价 | `close` | CNY/t |
| 持仓量 | `open_interest` | contracts |
| 成交量 | `volume` | contracts |
| Xinjiang553# / Kunming553# / Qinghai553# / InnerMongolia553# | `spot_553_xinjiang`, `spot_553_kunming`, ... | CNY/t |
| L1 / L2 closing price, open interest, volume | `nearby1_close`, `nearby2_close`, ... | CNY/t, contracts |

**Monthly indicators (SMM)**

| Source name | Column | Unit |
|---|---|---|
| SMM金属硅月度产量 | `mg_si_output` (metallurgical silicon output) | 10,000 t |
| 硅__97硅产量 | `si97_output` (97# silicon output) | 10,000 t |
| 硅__再生硅产量 | `recycled_si_output` | 10,000 t |
| SMM金属硅月度开工率 | `mg_si_operating_rate_pct` | % |
| 云南 / 四川 / 新疆样本周度开工率 | `yunnan_` / `sichuan_` / `xinjiang_weekly_operating_rate_pct` | % |
| SMM有机硅DMC月度产量 | `dmc_output` (organosilicon DMC) | 10,000 t |
| 有机硅DMC月度开工率 | `dmc_operating_rate_pct` | % |
| 原生铝合金月度产量：总量 | `primary_al_alloy_output` | 10,000 t |
| 再生铝合金月度产量 | `secondary_al_alloy_output` | 10,000 t |
| SMM多晶硅月度产量 | `polysilicon_output` | 10,000 t |
| 环比 / 同比 suffixes | `_mom_pct` / `_yoy_pct` (month-on-month / year-on-year) | % |

**Trade and inventory**

| Source name | Column | Unit |
|---|---|---|
| social_inventory_Mt | `social_inventory` (stocks in social warehouses) | 10,000 t |
| monthly_export_t / monthly_import_t | `exports_t` / `imports_t` | t |

## Translation of the original balance-sheet terms

| Chinese | English |
|---|---|
| 工业硅 / 金属硅 | Industrial silicon / metallurgical silicon (MG-Si) |
| 月度供需平衡表 | Monthly supply-demand balance |
| 本期供给 / 本期需求 | Supply / demand for the period |
| 供需差异 | Supply-demand gap (surplus) |
| 社库 (社会库存) | Social inventory |
| 期初库存 / 期末库存 | Opening / closing inventory |
| 有机硅 / 多晶硅 / 原生铝合金 | Organosilicon / polysilicon / primary aluminum alloy |
| 活跃合约 | Active (most-traded) contract |
| 开工率 | Operating rate |
| 丰水期 / 枯水期 | Wet season / dry season (hydropower) |
