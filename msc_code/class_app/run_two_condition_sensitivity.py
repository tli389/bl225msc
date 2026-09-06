"""Compare CLASS outcomes conditional on unemployment and HPI growth."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _local import (
    CLASS8_FEATURES,
    DEFAULT_HISTORIC_CSV,
    MinnesotaPosteriorVARGenerator,
    block_bridge_conditional_sample,
    clip_to_bounds,
    conditional_mixture_sample,
    load_us_class8_history,
    stable_seed,
)
from adapter import (
    build_class_macro_input,
    load_fed_class8_history,
    path_to_class8_frame,
)
from minneapolis_class import MinneapolisClassProjector
from run_generator_comparison import (
    BVAR_CASE,
    DEFAULT_ARCHIVE,
    DEFAULT_PROTOCOL,
    FUTURE_HORIZON,
    GAUSSIAN_GIB_COMPLETION_SEED,
    GENERATED_METHODS,
    GIB_GAUSSIAN_CASE,
    GIB_STUDENT_CASE,
    NO_LOWER_FLOOR_POLICY,
    Q0,
    TREASURY_RATE_FEATURES,
    _contact_mask,
    _fit_models,
    _json_ready,
    _load_domain,
    _operational_bounds,
    _projection_metrics,
    _selected_methods,
    _summarize,
    _support_warning_summary,
)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
# Read the matched-path and unemployment-only runs before this comparison.
DEFAULT_MATCHED_REFERENCE = (
    PROJECT_ROOT / "results" / "matched_official" / "matched_future_class8.csv"
)
DEFAULT_MATCHED_REFERENCE_MANIFEST = DEFAULT_MATCHED_REFERENCE.parent / "manifest.json"
DEFAULT_UNEMPLOYMENT_ONLY = PROJECT_ROOT / "results" / "generator_comparison"
GIB_CASE = GIB_STUDENT_CASE
CONDITIONED_FEATURES = ("unemployment", "hpi_qoq_growth")
SCENARIO_NAME = "fed_severe"


def _load_condition_reference(
    path: Path = DEFAULT_MATCHED_REFERENCE,
) -> pd.DataFrame:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"matched/rebased CLASS8 reference is missing: {path}")
    frame = pd.read_csv(path)
    expected_columns = ("quarter", *CLASS8_FEATURES)
    if tuple(frame.columns) != expected_columns:
        raise ValueError("matched/rebased reference does not use the CLASS8 schema")
    frame = frame.copy()
    frame.index = pd.PeriodIndex(frame.pop("quarter").astype(str), freq="Q")
    expected_index = pd.period_range(Q0 + 1, periods=FUTURE_HORIZON, freq="Q")
    if not frame.index.equals(expected_index):
        raise ValueError("matched/rebased reference does not cover 2020Q1--2023Q1")
    frame = frame.loc[:, list(CLASS8_FEATURES)].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame.to_numpy(dtype=float)).all():
        raise ValueError("matched/rebased reference contains non-finite values")
    return frame


def _prepare_path_banks(
    sampled_banks: Mapping[str, np.ndarray],
    reference: pd.DataFrame,
    *,
    n_paths: int,
    operational_bounds: Mapping[str, tuple[float, float]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, dict[str, float]]]:
    observed_indices = [CLASS8_FEATURES.index(name) for name in CONDITIONED_FEATURES]
    observed_values = reference.loc[:, list(CONDITIONED_FEATURES)].to_numpy(dtype=float)
    expected_shape = (n_paths, FUTURE_HORIZON, len(CLASS8_FEATURES))
    raw_banks: dict[str, np.ndarray] = {}
    operational_banks: dict[str, np.ndarray] = {}
    restoration: dict[str, dict[str, float]] = {}
    for method, sampled in sampled_banks.items():
        raw = np.asarray(sampled, dtype=float).copy()
        if raw.shape != expected_shape:
            raise RuntimeError(
                f"{method} returned shape {raw.shape}; expected {expected_shape}"
            )
        raw[:, :, observed_indices] = observed_values[None, :, :]
        if not np.isfinite(raw).all():
            raise RuntimeError(f"{method} returned non-finite raw paths")
        # Clip the CLASS input copy, then restore both supplied paths exactly.
        operational = clip_to_bounds(
            raw.copy(),
            CLASS8_FEATURES,
            operational_bounds,
        )
        operational[:, :, observed_indices] = observed_values[None, :, :]
        if not np.isfinite(operational).all():
            raise RuntimeError(f"{method} returned non-finite operational paths")
        raw_error = float(
            np.max(np.abs(raw[:, :, observed_indices] - observed_values[None, :, :]))
        )
        operational_error = float(
            np.max(
                np.abs(
                    operational[:, :, observed_indices]
                    - observed_values[None, :, :]
                )
            )
        )
        if raw_error != 0.0 or operational_error != 0.0:
            raise RuntimeError(f"{method} did not restore both supplied paths exactly")
        raw_banks[method] = raw
        operational_banks[method] = operational
        restoration[method] = {"raw": raw_error, "operational": operational_error}
    return raw_banks, operational_banks, restoration


def _sample_completions(
    gib: object,
    bvar: object,
    history: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    domain: dict[str, Any],
    protocol: dict[str, Any],
    smoke: bool,
    operational_bounds: Mapping[str, tuple[float, float]],
    gaussian_only: bool = False,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, object],
    dict[str, dict[str, float]],
]:
    student = protocol["student_t"]
    gib_config = protocol["gibvar"]
    n_paths = 48 if smoke else int(protocol["design"]["paths"])
    n_particles = 128 if smoke else int(student["conditional_particles"])
    state = history.iloc[-1].to_numpy(dtype=float)
    # Supply all 13 quarters of unemployment and matched HPI growth.
    observed = reference.loc[:, list(CONDITIONED_FEATURES)].to_numpy(dtype=float)

    sampled_banks: dict[str, np.ndarray] = {}
    diagnostics: dict[str, object] = {}
    if not gaussian_only:
        # The local Student-t backend uses resampling-only block conditioning.
        gib_paths, gib_diagnostics = block_bridge_conditional_sample(
            gib,
            observed,
            list(CONDITIONED_FEATURES),
            n_paths=n_paths,
            n_particles=n_particles,
            rng=np.random.default_rng(
                stable_seed(7, "class8_gib_completion", str(Q0))
            ),
            jumpoff=state,
            block_length=int(gib_config["block_length"]),
            tau=float(student["tau"]),
            kernel="student_t",
            degrees_of_freedom=float(student["degrees_of_freedom"]),
            ess_resample_ratio=float(student["ess_resample_ratio"]),
            likelihood_tempering=True,
            tempering_cess_ratio=float(student["tempering_cess_ratio"]),
            max_tempering_steps=(
                20 if smoke else int(student["max_tempering_steps"])
            ),
            rejuvenate_after_resampling=bool(student["rejuvenate_after_resampling"]),
            apply_hidden_bounds=False,
            features=CLASS8_FEATURES,
            bounds=domain["bounds"],
        )
        sampled_banks[GIB_CASE] = np.asarray(gib_paths, dtype=float)
        diagnostics[GIB_CASE] = gib_diagnostics

    supplied = np.zeros((FUTURE_HORIZON, len(CLASS8_FEATURES)), dtype=float)
    observed_indices = [CLASS8_FEATURES.index(name) for name in CONDITIONED_FEATURES]
    supplied[:, observed_indices] = observed
    gaussian_gib_paths, gaussian_gib_diagnostics = conditional_mixture_sample(
        gib,
        supplied,
        list(CONDITIONED_FEATURES),
        n_paths,
        np.random.default_rng(GAUSSIAN_GIB_COMPLETION_SEED),
        state,
        features=CLASS8_FEATURES,
        bounds=domain["bounds"],
        apply_hidden_bounds=False,
    )
    sampled_banks[GIB_GAUSSIAN_CASE] = np.asarray(
        gaussian_gib_paths,
        dtype=float,
    )
    diagnostics[GIB_GAUSSIAN_CASE] = gaussian_gib_diagnostics
    if not gaussian_only:
        # The BVAR conditions on the same supplied cells.
        bvar_paths, bvar_diagnostics = conditional_mixture_sample(
            bvar,
            supplied,
            list(CONDITIONED_FEATURES),
            n_paths,
            np.random.default_rng(stable_seed(7, "class8_bvar_completion", str(Q0))),
            state,
            features=CLASS8_FEATURES,
            bounds=domain["bounds"],
            apply_hidden_bounds=False,
        )
        sampled_banks[BVAR_CASE] = np.asarray(bvar_paths, dtype=float)
        diagnostics[BVAR_CASE] = bvar_diagnostics
    raw, operational, restoration = _prepare_path_banks(
        sampled_banks,
        reference,
        n_paths=n_paths,
        operational_bounds=operational_bounds,
    )
    return (
        raw,
        operational,
        diagnostics,
        restoration,
    )


def _bound_contact_fraction(
    values: np.ndarray,
    bounds: Mapping[str, tuple[float, float]],
) -> float:
    contacts = [
        _contact_mask(values[:, index], bounds[feature])
        for index, feature in enumerate(CLASS8_FEATURES)
        if feature not in CONDITIONED_FEATURES
    ]
    return float(np.concatenate(contacts).mean())


def _delta_against_unemployment_only(
    summary: pd.DataFrame,
    reference_summary: pd.DataFrame,
    *,
    methods: tuple[str, ...] = GENERATED_METHODS,
) -> pd.DataFrame:
    current = summary.set_index("method")
    previous = reference_summary.set_index("method")
    rows: list[dict[str, float | str]] = []
    for method in methods:
        if method not in current.index or method not in previous.index:
            raise ValueError(f"missing {method} from current or unemployment-only summary")
        median_columns = sorted(
            column
            for column in current.columns
            if column.endswith("_median") and column in previous.columns
        )
        for column in median_columns:
            current_value = float(current.loc[method, column])
            previous_value = float(previous.loc[method, column])
            rows.append(
                {
                    "method": method,
                    "metric": column.removesuffix("_median"),
                    "two_condition_median": current_value,
                    "unemployment_only_median": previous_value,
                    "delta": current_value - previous_value,
                }
            )
    return pd.DataFrame(rows)


def _write_results(
    output: Path,
    summary: pd.DataFrame,
    delta: pd.DataFrame,
    warning_summary: Mapping[str, object],
    *,
    smoke: bool,
    methods: tuple[str, ...] = GENERATED_METHODS,
) -> None:
    indexed = summary.set_index("method")
    lines = [
        "# Unemployment-plus-HPI CLASS sensitivity",
        "",
        "Condition: matched/rebased DFAST 2020 severely adverse unemployment and HPI q/q growth;",
        "2019Q4 bank state; no Treasury lower floor; fixed market and operational-loss overlays excluded.",
        "",
        "| Completion | Paths | Median minimum CET1 | Median cumulative NCO | Median cumulative PPNR |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in ("matched_rebased_fed_severe", *methods):
        row = indexed.loc[method]
        lines.append(
            f"| {method} | {int(row['n_paths'])} | "
            f"{float(row['minimum_cet1_ratio_pct_median']):.2f}% | "
            f"${float(row['cumulative_nco_bn_median']):.1f}bn | "
            f"${float(row['cumulative_ppnr_bn_median']):.1f}bn |"
        )
    lines.extend(["", "Median changes from the maintained unemployment-only run:", ""])
    for method in methods:
        subset = delta.loc[
            (delta["method"] == method)
            & delta["metric"].isin(
                ["minimum_cet1_ratio_pct", "cumulative_nco_bn", "cumulative_ppnr_bn"]
            )
        ]
        values = dict(zip(subset["metric"], subset["delta"], strict=True))
        lines.append(
            f"- {method}: minimum CET1 {values['minimum_cet1_ratio_pct']:+.3f} pp; "
            f"NCO {values['cumulative_nco_bn']:+.3f}bn; "
            f"PPNR {values['cumulative_ppnr_bn']:+.3f}bn."
        )
    lines.extend(["", "Support warnings:", ""])
    for method in methods:
        item = warning_summary[method]
        reasons = item["reasons"]
        text = "; ".join(reasons) if reasons else "none"
        lines.append(f"- {method}: {text}.")
    lines.extend(
        [
            "",
            (
                "This reduced smoke run checks wiring only."
                if smoke
                else "This is the production-sized post-development sensitivity."
            ),
            "It does not replace the maintained unemployment-only application or establish generator superiority.",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    output: Path,
    history_path: Path = DEFAULT_HISTORIC_CSV,
    protocol_path: Path = DEFAULT_PROTOCOL,
    archive: Path = DEFAULT_ARCHIVE,
    matched_reference_path: Path = DEFAULT_MATCHED_REFERENCE,
    matched_reference_manifest: Path = DEFAULT_MATCHED_REFERENCE_MANIFEST,
    unemployment_only_dir: Path = DEFAULT_UNEMPLOYMENT_ONLY,
    smoke: bool = True,
    gaussian_only: bool = False,
) -> Path:
    started = time.perf_counter()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    protocol, domain = _load_domain(Path(protocol_path))
    selected_methods = _selected_methods(gaussian_only)
    operational_bounds = _operational_bounds(domain, NO_LOWER_FLOOR_POLICY)
    # Estimate the macro models using history only through 2019Q4.
    model_history = load_us_class8_history(Path(history_path)).loc[:Q0].copy()
    actual_history = load_fed_class8_history(Path(history_path)).loc[:Q0].copy()
    if model_history.index.max() != Q0 or actual_history.index.max() != Q0:
        raise ValueError("CLASS8 histories must end at 2019Q4")
    reference = _load_condition_reference(Path(matched_reference_path))
    reference_output = reference.copy()
    reference_output.index.name = "quarter"
    reference_output.to_csv(output / "condition_reference.csv")

    matched_reference_manifest = Path(matched_reference_manifest).resolve()
    matched_manifest = json.loads(matched_reference_manifest.read_text(encoding="utf-8"))
    unemployment_only_dir = Path(unemployment_only_dir).resolve()
    unemployment_manifest_path = unemployment_only_dir / "manifest.json"
    unemployment_summary_path = unemployment_only_dir / "summary.csv"
    unemployment_manifest = json.loads(
        unemployment_manifest_path.read_text(encoding="utf-8")
    )
    if (
        unemployment_manifest.get("status") != "complete"
        or unemployment_manifest.get("treasury_rate_floor_policy", {}).get("mode")
        != NO_LOWER_FLOOR_POLICY
    ):
        raise ValueError("unemployment-only reference is not the completed no-floor run")
    unemployment_methods = set(unemployment_manifest.get("n_paths", {}))
    if unemployment_methods != set(selected_methods):
        raise ValueError(
            "unemployment-only reference does not match the selected generated methods"
        )
    unemployment_summary = pd.read_csv(unemployment_summary_path)
    if set(unemployment_summary["method"]).intersection(GENERATED_METHODS) != set(
        selected_methods
    ):
        raise ValueError(
            "unemployment-only summary is missing a generated completion method"
        )

    fit_started = time.perf_counter()
    gib, bvar, fit_seconds = _fit_models(
        model_history,
        domain=domain,
        protocol=protocol,
        smoke=smoke,
    )
    fit_elapsed = time.perf_counter() - fit_started
    completion_started = time.perf_counter()
    raw_banks, operational_banks, diagnostics, restoration = _sample_completions(
        gib,
        bvar,
        model_history,
        reference,
        domain=domain,
        protocol=protocol,
        smoke=smoke,
        operational_bounds=operational_bounds,
        gaussian_only=gaussian_only,
    )
    completion_elapsed = time.perf_counter() - completion_started
    np.savez_compressed(output / "raw_completed_paths.npz", **raw_banks)
    np.savez_compressed(output / "completed_paths.npz", **operational_banks)

    warning_summary = _support_warning_summary(diagnostics)
    support_payload = {
        "conditioned_features": list(CONDITIONED_FEATURES),
        "diagnostics": {
            method: _json_ready(value) for method, value in diagnostics.items()
        },
        "warning_summary": warning_summary,
    }
    (output / "support_diagnostics.json").write_text(
        json.dumps(_json_ready(support_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # Reuse one bank model for every completion and the matched complete path.
    projector = MinneapolisClassProjector.from_files(
        coefficients_path=Path(archive) / "output" / "estimated_model_coefficients.csv",
        y9c_path=Path(archive) / "y9c_bhc_data.csv",
        include_special_losses=False,
    )
    projection_started = time.perf_counter()
    rows: list[dict[str, float | int | str]] = []
    reference_macro = build_class_macro_input(
        actual_history,
        reference,
        class_q0=Q0,
        scenario_name="matched_rebased_fed_severe",
    )
    reference_metrics = _projection_metrics(
        projector.project_path(reference_macro),
        method="matched_rebased_fed_severe",
        path_id="matched_rebased_fed_severe",
    )
    for metric, expected in matched_manifest["metrics"].items():
        if metric in reference_metrics and not np.isclose(
            float(reference_metrics[metric]), float(expected), rtol=0.0, atol=1e-9
        ):
            raise RuntimeError(f"matched/rebased CLASS metric changed for {metric}")
    rows.append(reference_metrics)

    # Reconstruct CLASS yields and levels, then project each completed path.
    for method, paths in operational_banks.items():
        print(f"projecting {len(paths)} {method} paths through CLASS", flush=True)
        for index, values in enumerate(paths):
            macro = build_class_macro_input(
                actual_history,
                path_to_class8_frame(values, origin=Q0),
                class_q0=Q0,
                scenario_name=f"{method}_{index:04d}",
            )
            rows.append(
                _projection_metrics(
                    projector.project_path(macro),
                    method=method,
                    path_id=f"{method}_{index:04d}",
                    bound_contact_fraction=_bound_contact_fraction(
                        values,
                        operational_bounds,
                    ),
                )
            )
            if (index + 1) % 100 == 0 or index + 1 == len(paths):
                print(f"  {method}: projected {index + 1}/{len(paths)} paths", flush=True)
    projection_elapsed = time.perf_counter() - projection_started

    metrics = pd.DataFrame(rows)
    summary = _summarize(metrics)
    delta = _delta_against_unemployment_only(
        summary,
        unemployment_summary,
        methods=selected_methods,
    )
    metrics.to_csv(output / "path_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    delta.to_csv(output / "delta_vs_unemployment_only.csv", index=False)
    _write_results(
        output,
        summary,
        delta,
        warning_summary,
        smoke=smoke,
        methods=selected_methods,
    )

    n_paths = {method: int(len(paths)) for method, paths in operational_banks.items()}
    effective_rate_lowers = {
        feature: (
            None
            if not np.isfinite(operational_bounds[feature][0])
            else float(operational_bounds[feature][0])
        )
        for feature in TREASURY_RATE_FEATURES
    }
    manifest = {
        "status": "complete",
        "study": "isolated unemployment-plus-HPI CLASS8 sensitivity",
        "claim_eligible": False,
        "smoke": bool(smoke),
        "gaussian_only": bool(gaussian_only),
        "methods": list(selected_methods),
        "scenario": SCENARIO_NAME,
        "class_q0": str(Q0),
        "future_quarters": FUTURE_HORIZON,
        "reported_class_quarters": 9,
        "features": list(CLASS8_FEATURES),
        "conditioned_features": list(CONDITIONED_FEATURES),
        "n_paths": n_paths,
        "path_policy": (
            "raw conditional completions retained; operational copies have no "
            "Treasury lower floor, retain every other protocol bound, and restore "
            "both supplied paths exactly"
        ),
        "maximum_condition_restoration_error": restoration,
        "treasury_rate_floor_policy": {
            "mode": NO_LOWER_FLOOR_POLICY,
            "features": list(TREASURY_RATE_FEATURES),
            "protocol_lower_bounds": {
                feature: float(domain["bounds"][feature][0])
                for feature in TREASURY_RATE_FEATURES
            },
            "effective_lower_bounds": effective_rate_lowers,
            "upper_bounds_unchanged": {
                feature: float(operational_bounds[feature][1])
                for feature in TREASURY_RATE_FEATURES
            },
        },
        "special_loss_overlays": False,
        "condition_reference": {
            "scenario": SCENARIO_NAME,
            "matched_rebased": True,
            "source_path": str(Path(matched_reference_path).resolve()),
            "source_manifest_path": str(matched_reference_manifest),
            "future_start": str(reference.index.min()),
            "future_end": str(reference.index.max()),
            "transformations": {
                "unemployment": "direct published cpr_ur level",
                "hpi_qoq_growth": (
                    "100 * (HPI_t / HPI_{t-1} - 1), with 2019Q4 revised "
                    "history as the first-quarter base"
                ),
            },
        },
        "unemployment_only_reference": {
            "directory": str(unemployment_only_dir),
        },
        "simulation": {
            "paths_per_model": next(iter(n_paths.values())),
            **(
                {}
                if gaussian_only
                else {
                    "conditional_particles": (
                        128
                        if smoke
                        else int(protocol["student_t"]["conditional_particles"])
                    )
                }
            ),
            "gib_coefficient_systems": (
                4 if smoke else int(domain["gib_coefficient_systems"])
            ),
            **(
                {}
                if gaussian_only
                else {
                    "bvar_posterior_draws": (
                        4 if smoke else int(protocol["bvar"]["posterior_draws"])
                    )
                }
            ),
            "sampling_seed": 7,
            "sampling_seeds": {
                GIB_CASE: stable_seed(7, "class8_gib_completion", str(Q0)),
                GIB_GAUSSIAN_CASE: GAUSSIAN_GIB_COMPLETION_SEED,
                BVAR_CASE: stable_seed(7, "class8_bvar_completion", str(Q0)),
            }
            if not gaussian_only
            else {GIB_GAUSSIAN_CASE: GAUSSIAN_GIB_COMPLETION_SEED},
        },
        "timing_seconds": {
            "fit": float(fit_elapsed),
            "completion": float(completion_elapsed),
            "class_projection": float(projection_elapsed),
            "total": float(time.perf_counter() - started),
            "fit_by_method": (
                fit_seconds
                if not gaussian_only
                else {"gibvar": float(fit_seconds["gibvar"])}
            ),
        },
        "support_warning_summary": warning_summary,
        "limitations": [
            "This is a post-development sensitivity and does not replace the maintained unemployment-only application.",
            "The bank state and CLASS coefficients are fixed at 2019Q4.",
            "Satellite parameter uncertainty is not propagated.",
            "The supplied unemployment and HPI paths are hypothetical conditions, not realised labels.",
            "Support warnings identify concentration under the fitted approximation and are not plausibility probabilities.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(_json_ready(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORIC_CSV)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--matched-reference", type=Path, default=DEFAULT_MATCHED_REFERENCE)
    parser.add_argument(
        "--matched-reference-manifest",
        type=Path,
        default=DEFAULT_MATCHED_REFERENCE_MANIFEST,
    )
    parser.add_argument(
        "--unemployment-only-dir",
        type=Path,
        default=DEFAULT_UNEMPLOYMENT_ONLY,
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use 1,000 paths, 10,000 particles, 25 GIB systems and 100 BVAR draws.",
    )
    parser.add_argument(
        "--gaussian-only",
        action="store_true",
        help="Generate and project only Gaussian GIB paths plus the matched comparator.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    destination = run(
        output=args.out,
        history_path=args.history,
        protocol_path=args.protocol,
        archive=args.archive,
        matched_reference_path=args.matched_reference,
        matched_reference_manifest=args.matched_reference_manifest,
        unemployment_only_dir=args.unemployment_only_dir,
        smoke=not bool(args.full),
        gaussian_only=bool(args.gaussian_only),
    )
    print(f"wrote unemployment-plus-HPI CLASS sensitivity: {destination}", flush=True)


if __name__ == "__main__":
    main()


__all__ = [
    "CONDITIONED_FEATURES",
    "DEFAULT_MATCHED_REFERENCE",
    "DEFAULT_UNEMPLOYMENT_ONLY",
    "run",
]
