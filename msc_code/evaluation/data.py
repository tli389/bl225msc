"""data.py -- load the eight CLASS-aligned US variables from the Fed file.

Kept separate from the models so the code never mixes data construction with
estimation. UK data is licensed and not shipped; see the thesis package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = ["gdp_growth", "unemployment", "treasury_3m", "treasury_10y",
            "bbb_spread", "hpi_growth", "cre_growth", "equity_growth"]

# author-specified preferred graph, US system (identical to the thesis)
PREFERRED = {
    "gdp_growth":   ["unemployment", "treasury_3m", "bbb_spread"],
    "unemployment": ["gdp_growth", "bbb_spread"],
    "treasury_3m":  ["unemployment", "gdp_growth"],
    "treasury_10y": ["treasury_3m"],
    "bbb_spread":   ["unemployment", "gdp_growth"],
    "hpi_growth":   ["unemployment"],
    "cre_growth":   ["gdp_growth", "treasury_10y", "bbb_spread"],
    "equity_growth": ["gdp_growth", "bbb_spread"],
}


def load_us_history(path: str) -> pd.DataFrame:
    """Fed historic file -> eight variables, 1990Q1 onward."""
    raw = pd.read_csv(path)
    g = lambda c: pd.to_numeric(
        raw[c].astype(str).str.replace(",", "", regex=False),
        errors="coerce").to_numpy()          # .to_numpy(): stop index alignment
    growth = lambda c: pd.Series(g(c)).pct_change().to_numpy() * 100.0
    frame = pd.DataFrame({
        "gdp_growth":    g("Real GDP growth"),
        "unemployment":  g("Unemployment rate"),
        "treasury_3m":   g("3-month Treasury rate"),
        "treasury_10y":  g("10-year Treasury yield"),
        "bbb_spread":    g("BBB corporate yield") - g("10-year Treasury yield"),
        "hpi_growth":    growth("House Price Index (Level)"),
        "cre_growth":    growth("Commercial Real Estate Price Index (Level)"),
        "equity_growth": growth("Dow Jones Total Stock Market Index (Level)"),
    }, index=pd.PeriodIndex(raw["Date"].str.replace(" ", "", regex=False),
                            freq="Q"))
    frame = frame.loc["1990Q1":].dropna()
    assert not np.isnan(frame.to_numpy()).any()
    return frame


# ---- UK8 domain (locked uk8 protocol values) ----------------------------
UK_FEATURES = ["real_gdp_growth_annualized_pct", "unemployment_rate_pct",
               "bank_rate_pct", "gilt_10y_pct", "ig_corporate_spread_pct",
               "hpi_qoq_growth_pct", "equity_qoq_growth_pct",
               "cpi_inflation_yoy_pct"]

UK_PREFERRED = {
    "real_gdp_growth_annualized_pct": ["unemployment_rate_pct",
                                       "bank_rate_pct",
                                       "ig_corporate_spread_pct"],
    "unemployment_rate_pct": ["real_gdp_growth_annualized_pct",
                              "ig_corporate_spread_pct"],
    "bank_rate_pct": ["cpi_inflation_yoy_pct", "unemployment_rate_pct",
                      "real_gdp_growth_annualized_pct"],
    "gilt_10y_pct": ["bank_rate_pct", "cpi_inflation_yoy_pct",
                     "real_gdp_growth_annualized_pct"],
    "ig_corporate_spread_pct": ["equity_qoq_growth_pct",
                                "unemployment_rate_pct", "gilt_10y_pct"],
    "hpi_qoq_growth_pct": ["bank_rate_pct", "unemployment_rate_pct",
                           "real_gdp_growth_annualized_pct"],
    "equity_qoq_growth_pct": ["real_gdp_growth_annualized_pct",
                              "ig_corporate_spread_pct"],
    "cpi_inflation_yoy_pct": ["real_gdp_growth_annualized_pct",
                              "bank_rate_pct"],
}

UK_BOUNDS = {"real_gdp_growth_annualized_pct": (-100.0, 250.0),
             "unemployment_rate_pct": (0.0, 50.0),
             "bank_rate_pct": (-10.0, 50.0), "gilt_10y_pct": (-10.0, 50.0),
             "ig_corporate_spread_pct": (0.0, 50.0),
             "hpi_qoq_growth_pct": (-100.0, 250.0),
             "equity_qoq_growth_pct": (-100.0, 400.0),
             "cpi_inflation_yoy_pct": (-100.0, 100.0)}

UK_OWN_LAG = {"real_gdp_growth_annualized_pct": 0.0,
              "unemployment_rate_pct": 0.9, "bank_rate_pct": 0.9,
              "gilt_10y_pct": 0.9, "ig_corporate_spread_pct": 0.0,
              "hpi_qoq_growth_pct": 0.0, "equity_qoq_growth_pct": 0.0,
              "cpi_inflation_yoy_pct": 0.0}

def load_uk_history(path: str):
    """Licensed UK16 processed panel -> canonical UK8 view.

    The file is the provenance-tracked panel written by the research
    pipeline ('quarter' column + feature columns); it is licensed and never
    shipped.
    """
    raw = pd.read_csv(path)
    missing = {"quarter", *UK_FEATURES}.difference(raw.columns)
    if missing:
        raise ValueError(f"UK history is missing columns: {sorted(missing)}")
    frame = raw[UK_FEATURES].apply(pd.to_numeric, errors="raise")
    frame.index = pd.PeriodIndex(raw["quarter"].astype(str), freq="Q")
    frame = frame.sort_index()
    expected = pd.period_range(frame.index.min(), frame.index.max(), freq="Q")
    assert not frame.index.has_duplicates
    assert not len(expected.difference(frame.index)), "missing quarters"
    assert np.isfinite(frame.to_numpy(float)).all()
    return frame


def load_conditions_b(hist_path: str, cond_path: str):
    """Fed 2024 Exploratory Conditions B -> (history_df, target 12x8 array).
    Growth/spread construction as in the thesis: BBB spread = yield - 10y;
    index growth from published levels with the 2023Q4 historic level as base.
    """
    hist = load_us_history(hist_path)          # 1990Q1-2023Q4
    raw = pd.read_csv(cond_path)
    g = lambda c: pd.to_numeric(
        raw[c].astype(str).str.replace(",", "", regex=False),
        errors="coerce").to_numpy()
    hraw = pd.read_csv(hist_path)
    gh = lambda c: pd.to_numeric(
        hraw[c].astype(str).str.replace(",", "", regex=False),
        errors="coerce").to_numpy()
    def growth(col):
        levels = np.concatenate([[gh(col)[-1]], g(col)])   # 2023Q4 base
        return (levels[1:] / levels[:-1] - 1.0) * 100.0
    target = np.column_stack([
        g("Real GDP growth"), g("Unemployment rate"),
        g("3-month Treasury rate"), g("10-year Treasury yield"),
        g("BBB corporate yield") - g("10-year Treasury yield"),
        growth("House Price Index (Level)"),
        growth("Commercial Real Estate Price Index (Level)"),
        growth("Dow Jones Total Stock Market Index (Level)"),
    ])[:12]                                    # 2024Q1-2026Q4
    assert not np.isnan(target).any()
    return hist, target
