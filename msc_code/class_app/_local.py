"""Local wiring for the CLASS application.

The three run_*.py scripts were written inside the sealed research
package and imported their building blocks through package-relative
imports.  This module supplies the same names: the generators come from
../src through _backend.py, and the CLASS8 history loader is the
package's own, copied verbatim so the feature names match adapter.py and
data/class/class8_protocol.json.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from _backend import (  # noqa: E402
    GIBVAR, MinnesotaPosteriorVARGenerator, block_bridge_conditional_sample,
    clip_to_bounds, conditional_mixture_sample, stable_seed)

DEFAULT_HISTORIC_CSV = ROOT / "data" / "us" / "2026_Final_Historic_Domestic.csv"
DEFAULT_ARCHIVE = ROOT / "data" / "class" / "mpls_archive"
DEFAULT_PROTOCOL = ROOT / "data" / "class" / "class8_protocol.json"

# --- CLASS8 history loader (verbatim from the package's domains_class8) ---

CLASS8_HISTORY_START = pd.Period("1990Q1", freq="Q")

CLASS8_FEATURES = (
    "gdp_growth",
    "unemployment",
    "treasury_3m",
    "treasury_10y",
    "bbb_spread",
    "hpi_qoq_growth",
    "cre_qoq_growth",
    "equity_qoq_growth",
)

# Exact columns in the Federal Reserve domestic history and scenario files.
# Derived features list every raw input needed for their construction.
CLASS8_SOURCE_COLUMNS: Mapping[str, tuple[str, ...]] = {
    "gdp_growth": ("Real GDP growth",),
    "unemployment": ("Unemployment rate",),
    "treasury_3m": ("3-month Treasury rate",),
    "treasury_10y": ("10-year Treasury yield",),
    "bbb_spread": ("BBB corporate yield", "10-year Treasury yield"),
    "hpi_qoq_growth": ("House Price Index (Level)",),
    "cre_qoq_growth": ("Commercial Real Estate Price Index (Level)",),
    "equity_qoq_growth": ("Dow Jones Total Stock Market Index (Level)",),
}


def _numeric(raw: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(raw[column], errors="coerce")


def _percentage_growth(level: pd.Series, *, label: str) -> pd.Series:
    observed = level.dropna()
    if (observed <= 0.0).any():
        raise ValueError(f"{label} must be strictly positive for percentage growth")
    return level.pct_change(fill_method=None) * 100.0


def load_us_class8_history(path: Path) -> pd.DataFrame:
    """Load the eight CLASS-aligned Federal Reserve series from 1990Q1."""

    raw = pd.read_csv(path)
    required = {"Date"}
    for columns in CLASS8_SOURCE_COLUMNS.values():
        required.update(columns)
    missing = required.difference(raw.columns)
    if missing:
        raise ValueError(f"US CLASS8 history is missing columns: {sorted(missing)}")

    index = pd.PeriodIndex(
        raw["Date"].astype(str).str.strip().str.replace(" ", "", regex=False),
        freq="Q",
        name="quarter",
    )
    if index.has_duplicates:
        duplicates = index[index.duplicated()].unique().astype(str).tolist()
        raise ValueError(f"US CLASS8 history has duplicate quarters: {duplicates}")

    treasury_10y = _numeric(raw, "10-year Treasury yield")
    hpi_level = _numeric(raw, "House Price Index (Level)")
    cre_level = _numeric(raw, "Commercial Real Estate Price Index (Level)")
    equity_level = _numeric(raw, "Dow Jones Total Stock Market Index (Level)")
    frame = pd.DataFrame(
        {
            "gdp_growth": _numeric(raw, "Real GDP growth").to_numpy(),
            "unemployment": _numeric(raw, "Unemployment rate").to_numpy(),
            "treasury_3m": _numeric(raw, "3-month Treasury rate").to_numpy(),
            "treasury_10y": treasury_10y.to_numpy(),
            "bbb_spread": (
                _numeric(raw, "BBB corporate yield") - treasury_10y
            ).to_numpy(),
            "hpi_qoq_growth": _percentage_growth(
                hpi_level,
                label="House Price Index",
            ).to_numpy(),
            "cre_qoq_growth": _percentage_growth(
                cre_level,
                label="Commercial Real Estate Price Index",
            ).to_numpy(),
            "equity_qoq_growth": _percentage_growth(
                equity_level,
                label="Dow Jones Total Stock Market Index",
            ).to_numpy(),
        },
        index=index,
    )
    frame = frame.loc[CLASS8_HISTORY_START:].dropna(subset=list(CLASS8_FEATURES))
    frame = frame.sort_index()
    _validate_class8_history(frame)
    return frame.loc[:, list(CLASS8_FEATURES)]


def _validate_class8_history(history: pd.DataFrame) -> None:
    if tuple(history.columns) != CLASS8_FEATURES:
        raise ValueError("US CLASS8 feature order is not canonical")
    if not isinstance(history.index, pd.PeriodIndex):
        raise TypeError("US CLASS8 history must have a quarterly PeriodIndex")
    if history.index.has_duplicates or not history.index.is_monotonic_increasing:
        raise ValueError("US CLASS8 history index is not strictly chronological")
    expected = pd.period_range(history.index.min(), history.index.max(), freq="Q")
    missing = expected.difference(history.index)
    if len(missing):
        raise ValueError(
            "US CLASS8 history has missing quarters: "
            f"{missing.astype(str).tolist()}"
        )
    if not np.isfinite(history.to_numpy(dtype=float)).all():
        raise ValueError("US CLASS8 history contains non-finite values")


__all__ = [
    "CLASS8_FEATURES",
    "DEFAULT_ARCHIVE",
    "DEFAULT_HISTORIC_CSV",
    "DEFAULT_PROTOCOL",
    "GIBVAR",
    "MinnesotaPosteriorVARGenerator",
    "block_bridge_conditional_sample",
    "clip_to_bounds",
    "conditional_mixture_sample",
    "load_us_class8_history",
    "stable_seed",
]
