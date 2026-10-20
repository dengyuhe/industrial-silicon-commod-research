"""
Loading and cleaning the industrial silicon data, with Chinese source headers translated to English.

All raw files are vendor data (Wind, SMM) and are not distributed with this repository.
Place them in ``data/raw/`` under the file names listed in ``RAW_FILES`` (see data/README.md).
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

RAW_FILES = {
    "futures_active": "期货收盘价_活跃合约__工业硅.csv",      # Wind: active-contract close, OI, volume
    "spot_and_contracts": "Si553_price.xlsx",               # SMM spot 553# + contract-level futures
    "monthly_indicators": "ouput_monthly.xlsx",             # SMM monthly output / operating rates
    "other_metrics": "other_metrics.xlsx",                  # PV chain prices, inventory, trade
}

# --- Chinese indicator names (first line of each Wind/SMM header) -> English column names -------
INDICATORS = {
    "收盘价": "close",
    "持仓量": "open_interest",
    "成交量": "volume",
    "SMM金属硅月度产量": "mg_si_output",                       # 10k t
    "SMM金属硅月度产量：环比": "mg_si_output_mom_pct",
    "中国金属硅月度产量：同比": "mg_si_output_yoy_pct",
    "硅__97硅产量": "si97_output",                             # 10k t
    "硅__97硅产量：环比": "si97_output_mom_pct",
    "硅__再生硅产量": "recycled_si_output",                    # 10k t
    "硅__再生硅产量环比": "recycled_si_output_mom_pct",
    "云南样本周度开工率": "yunnan_weekly_operating_rate_pct",
    "四川样本周度开工率": "sichuan_weekly_operating_rate_pct",
    "新疆样本周度开工率": "xinjiang_weekly_operating_rate_pct",
    "四川样本周度开工率（大）": "sichuan_large_weekly_operating_rate_pct",
    "SMM金属硅月度开工率": "mg_si_operating_rate_pct",
    "SMM金属硅月度开工率：下月开工率E": "mg_si_operating_rate_next_month_est_pct",
    "SMM有机硅DMC月度产量": "dmc_output",                      # 10k t
    "有机硅DMC月度开工率": "dmc_operating_rate_pct",
    "SMM有机硅DMC下月排产量预计（月）-排产量": "dmc_output_next_month_plan",
    "原生铝合金月度产量：总量": "primary_al_alloy_output",       # 10k t
    "再生铝合金月度产量": "secondary_al_alloy_output",          # 10k t
    "SMM多晶硅月度产量": "polysilicon_output",                  # 10k t
    "SMM多晶硅月度产量环比": "polysilicon_output_mom_pct",
    "SMM多晶硅下月预计产量": "polysilicon_output_next_month_est",
    "硅__多晶硅月度产量": "polysilicon_output_alt",
    "硅__多晶硅月度产量：环比": "polysilicon_output_alt_mom_pct",
}

SPOT_COLUMNS = {
    "Xinjiang553#": "spot_553_xinjiang", "Qinghai553#": "spot_553_qinghai",
    "InnerMongolia553#": "spot_553_inner_mongolia", "Kunming553#": "spot_553_kunming",
    "L1closing_price": "nearby1_close", "L2closing_price": "nearby2_close",
    "L1open_interest": "nearby1_open_interest", "L2open_interest": "nearby2_open_interest",
    "L1volume": "nearby1_volume", "L2volume": "nearby2_volume",
}


def _english(header):
    """Map a multi-line Wind/SMM header ('name\\nunit\\nsource\\nid...') to an English name."""
    name = str(header).split("\n")[0].strip()
    return INDICATORS.get(name, name)


def _to_numeric(df):
    return df.replace({"-": np.nan, "": np.nan}).apply(pd.to_numeric, errors="coerce")


def load_futures(path=None):
    """Daily active-contract futures (CNY/t): close, open_interest, volume. Indexed by date."""
    path = Path(path) if path else RAW / RAW_FILES["futures_active"]
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns={"Date": "date"}).rename(columns=_english)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "close"])           # drops the trailing 'Source: Wind' footer
    return _to_numeric(df.set_index("date").sort_index())


def load_spot_and_contracts(path=None):
    """Daily 553# spot prices by region and first/second-nearby futures (CNY/t)."""
    path = Path(path) if path else RAW / RAW_FILES["spot_and_contracts"]
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").rename_axis("date").sort_index()
    return _to_numeric(df.rename(columns=SPOT_COLUMNS)[list(SPOT_COLUMNS.values())])


def load_monthly_indicators(path=None):
    """SMM monthly (and weekly) indicators, English column names. Output in 10,000 tonnes."""
    path = Path(path) if path else RAW / RAW_FILES["monthly_indicators"]
    df = pd.read_excel(path)
    df = df.rename(columns={df.columns[0]: "date"}).rename(columns=_english)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return _to_numeric(df.dropna(subset=["date"]).set_index("date").sort_index())


def load_trade_and_inventory(path=None):
    """Monthly exports/imports (tonnes) and weekly social inventory (10k t)."""
    path = Path(path) if path else RAW / RAW_FILES["other_metrics"]
    df = pd.read_excel(path)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").rename_axis("date").sort_index()
    return _to_numeric(df[["social_inventory_Mt", "monthly_export_t", "monthly_import_t"]]).rename(
        columns={"social_inventory_Mt": "social_inventory", "monthly_export_t": "exports_t",
                 "monthly_import_t": "imports_t"})


# --- Contract rolls --------------------------------------------------------------------------------
def roll_gaps(active, nearby, threshold=0.08):
    """
    Find days where the active-contract series jumps because it switched contracts.

    ``nearby`` is a DataFrame with first/second-nearby closes. On a day where the active close
    jumps by more than ``threshold`` while every individual contract moves far less, the jump is a
    roll artifact. Returns a DataFrame with the active return and the same-contract return.
    """
    j = active.to_frame("active").join(nearby[["nearby1_close", "nearby2_close"]], how="inner")
    r = j.pct_change()
    rows = []
    for d in r.index[r["active"].abs() > threshold]:
        prev = j.index[j.index.get_loc(d) - 1]
        # which contract is the active series on day d, and what did that contract do?
        leg = "nearby1_close" if np.isclose(j.at[d, "active"], j.at[d, "nearby1_close"]) else "nearby2_close"
        same = j.at[d, leg] / j.at[prev, leg] - 1
        if abs(same) < threshold / 2:
            rows.append(dict(date=d, active_return=r.at[d, "active"], same_contract_return=same,
                             prev_active_close=j.at[prev, "active"], prev_same_contract_close=j.at[prev, leg]))
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()


def roll_adjust(close, gaps, method="difference"):
    """
    Back-adjust prices so each roll contributes the same-contract move instead of the jump.

    Prices on and after the latest roll are unchanged. Earlier prices are shifted by the spread
    between the new and old contract on the day before the roll:
      method="difference": add the spread in CNY/t (keeps daily price changes in CNY intact;
                           appropriate for models of price levels such as ARIMA on prices)
      method="ratio":      multiply by the ratio of closes (keeps percentage returns intact;
                           appropriate for models of log returns)
    """
    adj = close.astype(float).copy()
    for d, g in gaps.sort_index(ascending=False).iterrows():
        before = adj.index < d
        if method == "difference":
            adj.loc[before] += g["prev_same_contract_close"] - g["prev_active_close"]
        elif method == "ratio":
            adj.loc[before] *= g["prev_same_contract_close"] / g["prev_active_close"]
        else:
            raise ValueError("method must be 'difference' or 'ratio'")
    return adj


# --- Monthly supply-demand balance -------------------------------------------------------------------
# Industrial silicon consumed per tonne of downstream output (from the report, Chapter 3)
SI_PER_POLYSILICON = 1.08     # t Si / t polysilicon
SI_PER_DMC = 0.52             # t Si / t DMC (2 t monomer per t DMC x 0.26 t Si per t monomer)
SI_PER_AL_ALLOY = 0.22        # t Si / t primary Al-Si alloy


def monthly_balance(indicators, trade):
    """
    Monthly supply, demand and surplus in 10,000 tonnes, following the report's balance sheet:
      supply = metallurgical Si + 97# Si + recycled Si + imports
      demand = 1.08 x polysilicon + 0.52 x DMC + 0.22 x primary Al alloy + exports
    """
    m = indicators.resample("ME").last()
    t = trade.resample("ME").last()
    out = pd.DataFrame(index=m.index)
    out["supply"] = (m["mg_si_output"] + m["si97_output"].fillna(0) + m["recycled_si_output"].fillna(0)
                     + t["imports_t"].fillna(0) / 1e4)
    out["demand_polysilicon"] = SI_PER_POLYSILICON * m["polysilicon_output"]
    out["demand_dmc"] = SI_PER_DMC * m["dmc_output"]
    out["demand_al_alloy"] = SI_PER_AL_ALLOY * m["primary_al_alloy_output"]
    out["exports"] = t["exports_t"] / 1e4
    out["demand"] = out[["demand_polysilicon", "demand_dmc", "demand_al_alloy", "exports"]].sum(axis=1, min_count=4)
    out["surplus"] = out["supply"] - out["demand"]
    return out.dropna(subset=["supply", "demand"])
