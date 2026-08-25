"""Convert CLASS8 path forecasts to the Minneapolis CLASS macro schema.

The statistical generators model three asset-price indexes as simple
quarter-on-quarter percentage changes.  The Minneapolis CLASS program instead
expects index levels and constructs its own log changes and year-on-year
changes.  This module performs that reversible boundary conversion while
leaving the supplied CLASS implementation untouched.

The supplied Minneapolis archive fixes its bank state at 2019Q4.  For that
reason ``class_q0`` is mandatory and future rows must begin in the immediately
following quarter.  The adapter never silently relabels a later forecast.
"""

from __future__ import annotations

import calendar
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

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

CLASS_MACRO_COLUMNS = (
    "type",
    "date",
    "dt",
    "growth",
    "cpr_ur",
    "cpr_t3m",
    "cpr_t10y",
    "bbb_corp",
    "cppi",
    "hpi",
    "djtsm",
)

SOURCE_LEVEL_COLUMNS = ("hpi", "cppi", "djtsm", "bbb_corp")
_LEVEL_GROWTH_PAIRS = (
    ("hpi_qoq_growth", "hpi"),
    ("cre_qoq_growth", "cppi"),
    ("equity_qoq_growth", "djtsm"),
)
_BACKWARD_ALIASES = {
    # The old US7/US16 implementation used this key for a series whose source
    # was actually the published three-month Treasury rate.
    "fed_funds": "treasury_3m",
    "short_rate": "treasury_3m",
}
_FED_COLUMNS = {
    "Date": "quarter",
    "Real GDP growth": "gdp_growth",
    "Unemployment rate": "unemployment",
    "3-month Treasury rate": "treasury_3m",
    "10-year Treasury yield": "treasury_10y",
    "BBB corporate yield": "bbb_corp",
    "House Price Index (Level)": "hpi",
    "Commercial Real Estate Price Index (Level)": "cppi",
    "Dow Jones Total Stock Market Index (Level)": "djtsm",
}


def _as_quarter(value: str | pd.Period) -> pd.Period:
    return pd.Period(value, freq="Q")


def _quarter_end_integer(quarter: pd.Period) -> int:
    month = int(quarter.end_time.month)
    day = calendar.monthrange(int(quarter.year), month)[1]
    return int(f"{quarter.year:04d}{month:02d}{day:02d}")


def _quarter_label(quarter: pd.Period) -> str:
    return f"{quarter.year}q{quarter.quarter}"


def _load_fed_raw(path: Path | str) -> pd.DataFrame:
    path = Path(path)
    raw = pd.read_csv(path)
    missing = set(_FED_COLUMNS).difference(raw.columns)
    if missing:
        raise ValueError(f"{path} is missing Fed columns: {sorted(missing)}")

    selected = raw[list(_FED_COLUMNS)].rename(columns=_FED_COLUMNS).copy()
    selected.index = pd.PeriodIndex(
        selected.pop("quarter")
        .astype(str)
        .str.strip()
        .str.replace(" ", "", regex=False),
        freq="Q",
        name="quarter",
    )
    for column in selected.columns:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    if selected.index.has_duplicates:
        duplicates = selected.index[selected.index.duplicated()].astype(str).tolist()
        raise ValueError(f"{path} contains duplicate quarters: {duplicates}")
    return selected.sort_index()


def _transform_fed_raw(raw: pd.DataFrame) -> pd.DataFrame:
    """Create the canonical CLASS8 state while preserving exact source levels."""
    result = raw.copy()
    result["bbb_spread"] = result["bbb_corp"] - result["treasury_10y"]
    for growth_column, level_column in _LEVEL_GROWTH_PAIRS:
        result[growth_column] = (
            result[level_column].pct_change(fill_method=None) * 100.0
        )
    ordered = list(CLASS8_FEATURES) + list(SOURCE_LEVEL_COLUMNS)
    return result[ordered]


def _validate_quarterly_index(frame: pd.DataFrame, label: str) -> None:
    if not isinstance(frame.index, pd.PeriodIndex):
        raise TypeError(f"{label} must use a quarterly PeriodIndex")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError(f"{label} must be strictly chronological and unique")
    if len(frame) > 1:
        expected = pd.period_range(frame.index.min(), frame.index.max(), freq="Q")
        missing = expected.difference(frame.index)
        if len(missing):
            raise ValueError(
                f"{label} has missing quarters: {missing.astype(str).tolist()}"
            )


def load_fed_class8_history(path: Path | str) -> pd.DataFrame:
    """Load a Fed historical file with CLASS8 features and source index levels.

    The canonical asset-price convention is simple quarter-on-quarter percent
    change, matching the maintained US16 loader.  Rows without all eight model
    variables are removed, so the joint history currently begins in 1988Q4.
    """
    result = _transform_fed_raw(_load_fed_raw(path))
    result = result.dropna(subset=list(CLASS8_FEATURES)).copy()
    _validate_quarterly_index(result, "Fed CLASS8 history")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Fed CLASS8 history contains non-finite values")
    return result


def load_fed_class8_scenario(
    scenario_path: Path | str,
    *,
    history_path: Path | str,
) -> pd.DataFrame:
    """Load a 13-quarter Fed scenario using the last actual level as its base.

    The first scenario-quarter growth for HPI, CPPI, and equities is computed
    against the immediately preceding actual quarter.  This makes the later
    recursive reconstruction exact rather than treating the first projected
    index level as an unrelated new base.
    """
    scenario_raw = _load_fed_raw(scenario_path)
    if len(scenario_raw) != 13:
        raise ValueError(
            f"Fed scenario must contain 13 quarters, found {len(scenario_raw)}"
        )
    _validate_quarterly_index(scenario_raw, "Fed scenario")

    origin = scenario_raw.index.min() - 1
    history_raw = _load_fed_raw(history_path)
    if origin not in history_raw.index:
        raise ValueError(
            f"Fed history does not contain the scenario base quarter {origin}"
        )
    combined = pd.concat([history_raw.loc[[origin]], scenario_raw])
    transformed = _transform_fed_raw(combined).loc[scenario_raw.index]
    if transformed[list(CLASS8_FEATURES)].isna().any().any():
        raise ValueError("Fed scenario contains missing CLASS8 inputs")
    if not np.isfinite(transformed.to_numpy(dtype=float)).all():
        raise ValueError("Fed scenario contains non-finite values")
    return transformed


def path_to_class8_frame(
    values: np.ndarray | Sequence[Sequence[float]],
    *,
    origin: str | pd.Period,
    feature_names: Sequence[str] = CLASS8_FEATURES,
) -> pd.DataFrame:
    """Attach canonical names and future quarters to one generated path."""
    array = np.asarray(values, dtype=float)
    names = tuple(feature_names)
    if array.ndim != 2:
        raise ValueError("one generated path must be a two-dimensional array")
    if array.shape != (13, len(names)):
        raise ValueError(
            "CLASS requires one 13-quarter path; "
            f"received shape {array.shape} for {len(names)} features"
        )
    q0 = _as_quarter(origin)
    index = pd.period_range(q0 + 1, periods=13, freq="Q", name="quarter")
    return pd.DataFrame(array, index=index, columns=names)


def _canonicalize_future(future: pd.DataFrame) -> pd.DataFrame:
    result = future.copy()
    for alias, canonical in _BACKWARD_ALIASES.items():
        if alias not in result.columns:
            continue
        if canonical in result.columns:
            left = pd.to_numeric(result[alias], errors="coerce").to_numpy(float)
            right = pd.to_numeric(result[canonical], errors="coerce").to_numpy(float)
            if not np.allclose(left, right, rtol=0.0, atol=1e-12, equal_nan=True):
                raise ValueError(f"conflicting {alias!r} and {canonical!r} columns")
        else:
            result = result.rename(columns={alias: canonical})

    missing = set(CLASS8_FEATURES).difference(result.columns)
    if missing:
        raise ValueError(f"future path is missing CLASS8 features: {sorted(missing)}")
    result = result[list(CLASS8_FEATURES)].copy()
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    _validate_quarterly_index(result, "future CLASS8 path")
    if len(result) != 13:
        raise ValueError(f"CLASS requires 13 future quarters, found {len(result)}")
    if not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("future CLASS8 path contains non-finite values")
    return result


def build_class_macro_input(
    actual_history: pd.DataFrame,
    future: pd.DataFrame,
    *,
    class_q0: str | pd.Period,
    scenario_name: str,
) -> pd.DataFrame:
    """Build the exact four-history plus thirteen-future CLASS input table.

    ``actual_history`` must include the four quarters ending at ``class_q0``
    and must retain the four source columns returned by
    :func:`load_fed_class8_history`.  ``future`` is one generated CLASS8 path.
    The function intentionally rejects any date mismatch instead of rebasing.
    """
    q0 = _as_quarter(class_q0)
    if not isinstance(scenario_name, str) or not scenario_name.strip():
        raise ValueError("scenario_name must be a non-empty string")
    if "," in scenario_name or "\n" in scenario_name or "\r" in scenario_name:
        raise ValueError("scenario_name cannot contain commas or line breaks")

    history = actual_history.copy()
    _validate_quarterly_index(history, "actual CLASS8 history")
    required_history = CLASS8_FEATURES + SOURCE_LEVEL_COLUMNS
    missing_history = set(required_history).difference(history.columns)
    if missing_history:
        raise ValueError(
            "actual history is missing CLASS8/source columns: "
            f"{sorted(missing_history)}"
        )
    history_quarters = pd.period_range(q0 - 3, q0, freq="Q")
    absent = history_quarters.difference(history.index)
    if len(absent):
        raise ValueError(
            "actual history is missing the four CLASS seed quarters: "
            f"{absent.astype(str).tolist()}"
        )
    history = history.loc[history_quarters, list(required_history)].copy()
    if not np.isfinite(history.to_numpy(dtype=float)).all():
        raise ValueError("the four CLASS seed quarters contain non-finite values")
    historical_spread = history["bbb_corp"] - history["treasury_10y"]
    if not np.allclose(
        historical_spread.to_numpy(dtype=float),
        history["bbb_spread"].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError(
            "actual history has inconsistent BBB yield, ten-year yield, and spread"
        )

    projected = _canonicalize_future(future)
    expected_future = pd.period_range(q0 + 1, periods=13, freq="Q")
    if not projected.index.equals(expected_future):
        raise ValueError(
            f"future path must run from {expected_future[0]} to "
            f"{expected_future[-1]} for class_q0={q0}; found "
            f"{projected.index[0]} to {projected.index[-1]}. "
            "The adapter does not silently rebase dates."
        )

    rows: list[dict[str, float | int | str]] = []
    for quarter, row in history.iterrows():
        rows.append(_class_row(scenario_name, quarter, row))

    current_levels = {
        level: float(history.iloc[-1][level]) for level in ("hpi", "cppi", "djtsm")
    }
    for quarter, row in projected.iterrows():
        for growth, level in _LEVEL_GROWTH_PAIRS:
            multiplier = 1.0 + float(row[growth]) / 100.0
            if multiplier <= 0.0:
                raise ValueError(
                    f"{growth}={row[growth]} at {quarter} implies a non-positive "
                    f"{level} level under the canonical simple-percent convention"
                )
            current_levels[level] *= multiplier
        augmented = row.copy()
        augmented["bbb_corp"] = float(row["bbb_spread"] + row["treasury_10y"])
        for level, value in current_levels.items():
            augmented[level] = value
        rows.append(_class_row(scenario_name, quarter, augmented))

    result = pd.DataFrame(rows, columns=CLASS_MACRO_COLUMNS)
    validate_class_macro_input(result, class_q0=q0)
    _validate_round_trip(result, projected)
    return result


def _class_row(
    scenario_name: str,
    quarter: pd.Period,
    row: pd.Series,
) -> dict[str, float | int | str]:
    return {
        "type": scenario_name,
        "date": _quarter_end_integer(quarter),
        "dt": _quarter_label(quarter),
        "growth": float(row["gdp_growth"]),
        "cpr_ur": float(row["unemployment"]),
        "cpr_t3m": float(row["treasury_3m"]),
        "cpr_t10y": float(row["treasury_10y"]),
        "bbb_corp": float(row["bbb_corp"]),
        "cppi": float(row["cppi"]),
        "hpi": float(row["hpi"]),
        "djtsm": float(row["djtsm"]),
    }


def _periods_from_dates(values: pd.Series) -> pd.PeriodIndex:
    periods: list[pd.Period] = []
    for raw_value in values:
        text = str(int(raw_value))
        if len(text) != 8:
            raise ValueError(f"CLASS date must be YYYYMMDD, found {raw_value!r}")
        timestamp = pd.Timestamp(
            year=int(text[:4]), month=int(text[4:6]), day=int(text[6:8])
        )
        quarter = timestamp.to_period("Q")
        if int(text) != _quarter_end_integer(quarter):
            raise ValueError(f"CLASS date is not a calendar quarter end: {text}")
        periods.append(quarter)
    return pd.PeriodIndex(periods, freq="Q", name="quarter")


def validate_class_macro_input(
    frame: pd.DataFrame,
    *,
    class_q0: str | pd.Period,
) -> None:
    """Validate the exact schema and lag support consumed by the R program."""
    if tuple(frame.columns) != CLASS_MACRO_COLUMNS:
        raise ValueError(
            "CLASS macro columns must be exactly "
            f"{list(CLASS_MACRO_COLUMNS)}, found {list(frame.columns)}"
        )
    if len(frame) != 17:
        raise ValueError(f"CLASS input must have 17 rows (4 + 13), found {len(frame)}")

    q0 = _as_quarter(class_q0)
    quarters = _periods_from_dates(frame["date"])
    expected = pd.period_range(q0 - 3, periods=17, freq="Q")
    if not quarters.equals(expected):
        raise ValueError(
            f"CLASS rows must be consecutive from {expected[0]} to {expected[-1]}"
        )
    expected_dt = [_quarter_label(quarter) for quarter in expected]
    if frame["dt"].astype(str).tolist() != expected_dt:
        raise ValueError("CLASS dt labels do not match the date column")
    scenario_types = frame["type"].astype(str).str.strip().unique()
    if len(scenario_types) != 1 or not scenario_types[0]:
        raise ValueError("CLASS input must contain one non-empty scenario type")

    numeric_columns = [
        column for column in CLASS_MACRO_COLUMNS if column not in {"type", "dt"}
    ]
    numeric = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("CLASS macro input contains non-finite numeric values")
    if (numeric[["cppi", "hpi", "djtsm"]] <= 0.0).any().any():
        raise ValueError("CLASS index levels must be strictly positive")

    predictors = derive_class_predictors(frame)
    future_predictors = predictors.iloc[4:]
    if not np.isfinite(future_predictors.to_numpy(dtype=float)).all():
        raise ValueError(
            "four historical rows do not provide complete lags for all 13 future rows"
        )


def derive_class_predictors(frame: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the macro transformations performed in the supplied R file."""
    values = frame.copy()
    result = pd.DataFrame(index=values.index)
    result["spread"] = values["cpr_t10y"] - values["cpr_t3m"]
    result["d10y"] = values["cpr_t10y"].diff()
    result["stockgrowth"] = np.log(values["djtsm"] / values["djtsm"].shift(1)) * 100.0
    result["unchange"] = values["cpr_ur"].diff() * 4.0
    result["yyhpi"] = np.log(values["hpi"] / values["hpi"].shift(4)) * 100.0
    result["yyhpi_zero"] = np.minimum(result["yyhpi"], 0.0)
    result["yycppi"] = np.log(values["cppi"] / values["cppi"].shift(4)) * 100.0
    result["yycppi_zero"] = np.minimum(result["yycppi"], 0.0)
    result["bspread"] = values["bbb_corp"] - values["cpr_t10y"]
    result["dbspread"] = result["bspread"].diff()
    result["dbspread_pos"] = np.maximum(result["dbspread"], 0.0)
    return result


def _validate_round_trip(output: pd.DataFrame, future: pd.DataFrame) -> None:
    future_output = output.iloc[4:].reset_index(drop=True)
    base_and_future = output.iloc[3:].reset_index(drop=True)
    for growth, level in _LEVEL_GROWTH_PAIRS:
        recovered = (
            base_and_future[level].iloc[1:].to_numpy(dtype=float)
            / base_and_future[level].iloc[:-1].to_numpy(dtype=float)
            - 1.0
        ) * 100.0
        expected = future[growth].to_numpy(dtype=float)
        if not np.allclose(recovered, expected, rtol=1e-11, atol=1e-11):
            raise AssertionError(f"{level} reconstruction failed its round-trip check")
    recovered_spread = future_output["bbb_corp"].to_numpy(dtype=float) - future_output[
        "cpr_t10y"
    ].to_numpy(dtype=float)
    if not np.allclose(
        recovered_spread,
        future["bbb_spread"].to_numpy(dtype=float),
        rtol=1e-12,
        atol=1e-12,
    ):
        raise AssertionError("BBB-yield reconstruction failed its round-trip check")


def write_class_macro_input(
    frame: pd.DataFrame,
    path: Path | str,
    *,
    class_q0: str | pd.Period,
    overwrite: bool = False,
) -> Path:
    """Validate and write one CLASS scenario without silently overwriting it."""
    destination = Path(path)
    validate_class_macro_input(frame, class_q0=class_q0)
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite existing CLASS input: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    return destination
