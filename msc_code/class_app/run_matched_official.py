"""Project a Fed severe path through CLASS with the generator seed history.

The generator comparison prepends revised Fed observations for 2019Q1--Q4 to
every generated path.  The Minneapolis archive's published scenario instead
contains its own, older 2019 history.  This module constructs a separately
labelled official-path comparator with the same revised seed history and the
same frozen CLASS settings as the generated paths.

The default rebase_growth policy preserves the archive scenario's future
quarter-on-quarter HPI, commercial-property and equity-index movements, then
rebases those movements onto the revised 2019Q4 index levels.  This matches
the growth-rate state supplied by the statistical generators.  The optional
exact_levels policy retains the archive's future index levels literally;
because its 2019Q4 levels differ from the revised history, this creates a
revision-driven jump at the splice and is reported only as a sensitivity.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _local import DEFAULT_ARCHIVE, DEFAULT_HISTORIC_CSV
from adapter import (
    CLASS8_FEATURES,
    build_class_macro_input,
    load_fed_class8_history,
)
from minneapolis_class import ClassProjection, MinneapolisClassProjector

HERE = Path(__file__).resolve().parent
Q0 = pd.Period("2019Q4", freq="Q")
FUTURE_HORIZON = 13
ASSET_PAIRS = (
    ("hpi_qoq_growth", "hpi"),
    ("cre_qoq_growth", "cppi"),
    ("equity_qoq_growth", "djtsm"),
)


def _load_archive_scenario(archive: Path, scenario_name: str) -> pd.DataFrame:
    raw = pd.read_csv(archive / "macro_data_proj.csv")
    selected = raw.loc[raw["type"].astype(str).eq(scenario_name)].copy()
    if selected.empty:
        choices = sorted(raw["type"].dropna().astype(str).unique())
        raise ValueError(f"unknown CLASS scenario {scenario_name!r}; choose from {choices}")
    selected["date"] = pd.to_numeric(selected["date"], errors="raise").astype(int)
    selected = selected.sort_values("date").reset_index(drop=True)
    q0_rows = selected.loc[selected["date"].eq(20191231)]
    future = selected.loc[selected["date"].gt(20191231)]
    if len(q0_rows) != 1 or len(future) != FUTURE_HORIZON:
        raise ValueError(
            f"{scenario_name} must contain one 2019Q4 row and 13 future rows"
        )
    return selected


def _future_class8(
    scenario: pd.DataFrame,
    revised_history: pd.DataFrame,
    *,
    asset_level_policy: str,
) -> pd.DataFrame:
    q0_row = scenario.loc[scenario["date"].eq(20191231)].iloc[0]
    future_raw = scenario.loc[scenario["date"].gt(20191231)].copy()
    future_raw["quarter"] = pd.PeriodIndex(
        pd.to_datetime(future_raw["date"].astype(str), format="%Y%m%d"),
        freq="Q",
    )
    future_raw = future_raw.set_index("quarter")

    future = pd.DataFrame(index=future_raw.index)
    future["gdp_growth"] = future_raw["growth"].to_numpy(dtype=float)
    future["unemployment"] = future_raw["cpr_ur"].to_numpy(dtype=float)
    future["treasury_3m"] = future_raw["cpr_t3m"].to_numpy(dtype=float)
    future["treasury_10y"] = future_raw["cpr_t10y"].to_numpy(dtype=float)
    future["bbb_spread"] = (
        future_raw["bbb_corp"].to_numpy(dtype=float)
        - future_raw["cpr_t10y"].to_numpy(dtype=float)
    )

    if asset_level_policy not in {"rebase_growth", "exact_levels"}:
        raise ValueError("asset_level_policy must be 'rebase_growth' or 'exact_levels'")
    for growth_name, level_name in ASSET_PAIRS:
        if asset_level_policy == "rebase_growth":
            base = float(q0_row[level_name])
        else:
            base = float(revised_history.loc[Q0, level_name])
        levels = np.r_[base, future_raw[level_name].to_numpy(dtype=float)]
        future[growth_name] = (levels[1:] / levels[:-1] - 1.0) * 100.0

    future = future[list(CLASS8_FEATURES)]
    if not np.isfinite(future.to_numpy(dtype=float)).all():
        raise ValueError("matched official future contains non-finite values")
    return future


def _projection_metrics(projection: ClassProjection) -> dict[str, float | int]:
    aggregate = projection.aggregate.sort_values("date")
    future_dates = set(aggregate["date"].iloc[1:].astype(int))
    detail = projection.detail.loc[projection.detail["date"].astype(int).isin(future_dates)]
    return {
        "minimum_cet1_ratio_pct": float(aggregate["cap_ratio"].min()),
        "end_cet1_ratio_pct": float(aggregate["cap_ratio"].iloc[-1]),
        "maximum_cet1_drawdown_bn": float(-aggregate["cet1_drop"].min()),
        "cumulative_nco_bn": float(detail["qnetxoff_total"].sum() / 1_000_000.0),
        "cumulative_provisions_bn": float(detail["qprov"].sum() / 1_000_000.0),
        "cumulative_ppnr_bn": float(detail["ppnr"].sum() / 1_000_000.0),
        "cumulative_net_income_bn": float(detail["netinc"].sum() / 1_000_000.0),
        "firms_breaching": int(projection.breaches["breach"].sum()),
    }


def _future_match_diagnostics(
    macro: pd.DataFrame,
    archive_scenario: pd.DataFrame,
) -> dict[str, Any]:
    archive_future = archive_scenario.loc[
        archive_scenario["date"].gt(20191231)
    ].reset_index(drop=True)
    generated_future = macro.iloc[4:].reset_index(drop=True)
    exact_columns = [
        "growth",
        "cpr_ur",
        "cpr_t3m",
        "cpr_t10y",
        "bbb_corp",
    ]
    exact_error = np.max(
        np.abs(
            generated_future[exact_columns].to_numpy(dtype=float)
            - archive_future[exact_columns].to_numpy(dtype=float)
        )
    )
    scale_ratios = {
        level: float(generated_future[level].iloc[0] / archive_future[level].iloc[0])
        for level in ("hpi", "cppi", "djtsm")
    }
    level_errors = {
        level: float(
            np.max(
                np.abs(
                    generated_future[level].to_numpy(dtype=float)
                    - archive_future[level].to_numpy(dtype=float)
                )
            )
        )
        for level in ("hpi", "cppi", "djtsm")
    }
    return {
        "maximum_exact_non_index_future_error": float(exact_error),
        "future_index_first_quarter_scale_ratio_to_archive": scale_ratios,
        "maximum_future_index_level_error_to_archive": level_errors,
    }


def run(
    *,
    output: Path,
    history_path: Path = DEFAULT_HISTORIC_CSV,
    archive: Path = DEFAULT_ARCHIVE,
    scenario_name: str = "fed_severe",
    asset_level_policy: str = "rebase_growth",
) -> Path:
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    archive = Path(archive).resolve()
    history_path = Path(history_path).resolve()
    revised_history = load_fed_class8_history(history_path).loc[:Q0].copy()
    if revised_history.index.max() != Q0:
        raise ValueError("revised Fed history does not end at 2019Q4")
    scenario = _load_archive_scenario(archive, scenario_name)
    future = _future_class8(
        scenario,
        revised_history,
        asset_level_policy=asset_level_policy,
    )
    macro = build_class_macro_input(
        revised_history,
        future,
        class_q0=Q0,
        scenario_name=f"matched_{scenario_name}_{asset_level_policy}",
    )

    projector = MinneapolisClassProjector.from_files(
        coefficients_path=archive / "output" / "estimated_model_coefficients.csv",
        y9c_path=archive / "y9c_bhc_data.csv",
        include_special_losses=False,
    )
    projection = projector.project_path(macro)
    metrics = _projection_metrics(projection)
    diagnostics = _future_match_diagnostics(macro, scenario)

    macro.to_csv(output / "matched_macro_input.csv", index=False)
    future.to_csv(output / "matched_future_class8.csv", index=True)
    pd.DataFrame([metrics]).to_csv(output / "metrics.csv", index=False)
    projection.aggregate.to_csv(output / "aggregate_projection.csv", index=False)
    projection.firm.to_csv(output / "firm_projection.csv", index=False)
    projection.breaches.to_csv(output / "firm_breaches.csv", index=False)
    projection.detail.to_csv(output / "projection_detail.csv", index=False)

    manifest = {
        "status": "complete",
        "scenario": scenario_name,
        "class_q0": str(Q0),
        "seed_history": "revised Fed observations, 2019Q1--2019Q4",
        "asset_level_policy": asset_level_policy,
        "special_loss_overlays": False,
        "fixed_class_coefficients_and_bank_state": True,
        "metrics": metrics,
        "match_diagnostics": diagnostics,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORIC_CSV)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--scenario", default="fed_severe")
    parser.add_argument(
        "--asset-level-policy",
        choices=("rebase_growth", "exact_levels"),
        default="rebase_growth",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = run(
        output=args.out,
        history_path=args.history,
        archive=args.archive,
        scenario_name=str(args.scenario),
        asset_level_policy=str(args.asset_level_policy),
    )
    print(f"wrote matched official CLASS projection: {output}", flush=True)


if __name__ == "__main__":
    main()
