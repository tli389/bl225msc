"""Run a CLASS8 generator-to-capital comparison at the 2019Q4 snapshot.

This pilot fits GIB--VAR and the Minnesota BVAR to the CLASS-aligned eight-
variable history available at 2019Q4.  Both models receive the same future
unemployment path from one Minneapolis scenario (``fed_severe`` by default)
and complete the other seven variables for 13 quarters.  A declared
application policy is then applied before the paths enter one frozen,
fit-once Minneapolis CLASS projector.  The optional ``no-lower-floor`` policy
allows negative Treasury rates while preserving every other configured bound.

The fixed DFAST 2020 operational-risk and market-shock overlays are excluded
by default so differences arise from the completed macroeconomic paths.  The
output is exploratory and conditional on the 2019Q4 bank state and published
CLASS coefficients; it is not a current bank-capital forecast.

``--gaussian-only`` generates only Gaussian GIB--VAR paths plus the official
scenario comparator, allowing the sealed Student-t and BVAR results to remain
unchanged.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from _local import (
    CLASS8_FEATURES,
    DEFAULT_ARCHIVE,
    DEFAULT_HISTORIC_CSV,
    DEFAULT_PROTOCOL,
    GIBVAR,
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
from minneapolis_class import ClassProjection, MinneapolisClassProjector

HERE = Path(__file__).resolve().parent
Q0 = pd.Period("2019Q4", freq="Q")
FUTURE_HORIZON = 13
TREASURY_RATE_FEATURES = ("treasury_3m", "treasury_10y")
PROTOCOL_FLOOR_POLICY = "protocol"
NO_LOWER_FLOOR_POLICY = "no-lower-floor"
TREASURY_FLOOR_POLICIES = (PROTOCOL_FLOOR_POLICY, NO_LOWER_FLOOR_POLICY)
GIB_STUDENT_CASE = "gibvar_student_t"
GIB_GAUSSIAN_CASE = "gibvar_gaussian"
BVAR_CASE = "minnesota_bvar"
GENERATED_METHODS = (GIB_STUDENT_CASE, GIB_GAUSSIAN_CASE, BVAR_CASE)
GAUSSIAN_GIB_COMPLETION_SEED = stable_seed(
    7,
    "class8_gib_gaussian_completion",
    str(Q0),
)


def _selected_methods(gaussian_only: bool) -> tuple[str, ...]:
    return (GIB_GAUSSIAN_CASE,) if gaussian_only else GENERATED_METHODS


def _json_ready(value: object) -> object:
    """Convert nested completion diagnostics to strict JSON values."""
    if is_dataclass(value):
        return _json_ready(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _support_warning_summary(
    diagnostics: Mapping[str, object],
) -> dict[str, object]:
    """Apply the maintained method-specific conditional-support warnings."""
    summary: dict[str, object] = {}
    for method, diagnostic in diagnostics.items():
        if method == GIB_STUDENT_CASE:
            values = {
                "minimum_ess_ratio": float(diagnostic.minimum_ess_ratio),
                "effective_components": float(diagnostic.effective_components),
                "maximum_component_share": float(
                    diagnostic.maximum_component_share
                ),
                "minimum_effective_selected_blocks": float(
                    np.min(diagnostic.effective_selected_blocks)
                ),
                "maximum_selected_block_share": float(
                    np.max(diagnostic.maximum_selected_block_share)
                ),
                "unique_ancestor_ratio": float(diagnostic.unique_ancestor_ratio),
            }
            thresholds = {
                "minimum_ess_ratio": 0.10,
                "minimum_effective_components": 5.0,
                "minimum_effective_selected_blocks": 5.0,
                "maximum_component_share": 0.50,
                "maximum_selected_block_share": 0.50,
                "minimum_unique_ancestor_ratio": 0.01,
            }
            reasons: list[str] = []
            if values["minimum_ess_ratio"] < thresholds["minimum_ess_ratio"]:
                reasons.append("minimum ESS ratio below 0.10")
            if (
                values["effective_components"]
                < thresholds["minimum_effective_components"]
            ):
                reasons.append("fewer than five effective fitted components")
            if (
                values["minimum_effective_selected_blocks"]
                < thresholds["minimum_effective_selected_blocks"]
            ):
                reasons.append("fewer than five effective selected block centres")
            if (
                values["maximum_component_share"]
                > thresholds["maximum_component_share"]
            ):
                reasons.append("maximum fitted-component share above 0.50")
            if (
                values["maximum_selected_block_share"]
                > thresholds["maximum_selected_block_share"]
            ):
                reasons.append("maximum selected-block share above 0.50")
            if (
                values["unique_ancestor_ratio"]
                < thresholds["minimum_unique_ancestor_ratio"]
            ):
                reasons.append("unique-ancestor ratio below 0.01")
        elif method in {GIB_GAUSSIAN_CASE, BVAR_CASE}:
            minimum_effective = max(1.5, 0.10 * int(diagnostic.component_count))
            values = {
                "component_count": int(diagnostic.component_count),
                "effective_components": float(diagnostic.effective_components),
                "maximum_weight": float(diagnostic.maximum_weight),
                "minimum_mahalanobis_per_dimension": float(
                    diagnostic.minimum_mahalanobis_per_dimension
                ),
            }
            thresholds = {
                "minimum_effective_components": float(minimum_effective),
                "maximum_weight": 0.90,
                "maximum_mahalanobis_per_dimension": 4.0,
            }
            reasons = []
            if values["effective_components"] < minimum_effective:
                reasons.append("effective component count below declared threshold")
            if values["maximum_weight"] > thresholds["maximum_weight"]:
                reasons.append("maximum component weight above 0.90")
            if (
                values["minimum_mahalanobis_per_dimension"]
                > thresholds["maximum_mahalanobis_per_dimension"]
            ):
                reasons.append("minimum supplied-path distance above 4")
        else:
            raise ValueError(f"unsupported CLASS completion method: {method}")
        summary[method] = {
            "triggered": bool(reasons),
            "reasons": reasons,
            "thresholds": thresholds,
            "values": values,
        }
    return summary


def _operational_bounds(
    domain: Mapping[str, Any],
    treasury_floor_policy: str,
) -> dict[str, tuple[float, float]]:
    """Return a copied application-bound map under the selected rate policy."""
    if treasury_floor_policy not in TREASURY_FLOOR_POLICIES:
        raise ValueError(
            "treasury_floor_policy must be one of "
            f"{TREASURY_FLOOR_POLICIES}, found {treasury_floor_policy!r}"
        )
    bounds = {
        str(feature): (float(values[0]), float(values[1]))
        for feature, values in domain["bounds"].items()
    }
    if treasury_floor_policy == NO_LOWER_FLOOR_POLICY:
        for feature in TREASURY_RATE_FEATURES:
            _, upper = bounds[feature]
            bounds[feature] = (-np.inf, upper)
    return bounds


def _contact_mask(
    values: np.ndarray,
    limits: tuple[float, float],
) -> np.ndarray:
    """Identify finite lower- or upper-bound contacts."""
    lower, upper = limits
    contacts = np.zeros(np.asarray(values).shape, dtype=bool)
    if np.isfinite(lower):
        contacts |= np.isclose(values, lower, rtol=0.0, atol=1e-12)
    if np.isfinite(upper):
        contacts |= np.isclose(values, upper, rtol=0.0, atol=1e-12)
    return contacts


def _prepare_path_banks(
    sampled_banks: Mapping[str, np.ndarray],
    unemployment: np.ndarray,
    *,
    n_paths: int,
    operational_bounds: Mapping[str, tuple[float, float]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Validate raw completions and create separate operational copies."""
    unemployment_index = CLASS8_FEATURES.index("unemployment")
    raw_banks: dict[str, np.ndarray] = {}
    operational_banks: dict[str, np.ndarray] = {}
    expected_shape = (n_paths, FUTURE_HORIZON, len(CLASS8_FEATURES))
    for name, sampled in sampled_banks.items():
        raw = np.asarray(sampled, dtype=float).copy()
        if raw.shape != expected_shape:
            raise RuntimeError(
                f"{name} returned shape {raw.shape}; expected {expected_shape}"
            )
        raw[:, :, unemployment_index] = unemployment[None, :]
        if not np.isfinite(raw).all():
            raise RuntimeError(f"{name} returned non-finite raw paths")
        raw_banks[name] = raw

        operational = clip_to_bounds(
            raw.copy(),
            CLASS8_FEATURES,
            operational_bounds,
        )
        operational[:, :, unemployment_index] = unemployment[None, :]
        if not np.isfinite(operational).all():
            raise RuntimeError(f"{name} returned non-finite operational paths")
        operational_banks[name] = operational
    return raw_banks, operational_banks


def _load_domain(protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    domain = protocol["domains"]["us_class8"]
    if tuple(domain["features"]) != CLASS8_FEATURES:
        raise ValueError("CLASS8 protocol feature order differs from the loader")
    return protocol, domain


def _official_condition(
    archive: Path,
    scenario_name: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    scenarios = pd.read_csv(archive / "macro_data_proj.csv")
    selected = scenarios.loc[
        scenarios["type"].astype(str).eq(scenario_name)
    ].copy()
    if selected.empty:
        choices = sorted(scenarios["type"].dropna().astype(str).unique())
        raise ValueError(f"unknown CLASS scenario {scenario_name!r}; choose from {choices}")
    selected["date"] = pd.to_numeric(selected["date"], errors="raise").astype(int)
    future = selected.loc[selected["date"] > 20191231].sort_values("date")
    unemployment = future["cpr_ur"].to_numpy(dtype=float)
    if len(unemployment) != FUTURE_HORIZON or not np.isfinite(unemployment).all():
        raise ValueError(
            f"{scenario_name} must provide 13 finite future unemployment values"
        )
    return selected, unemployment


def _fit_models(
    history: pd.DataFrame,
    *,
    domain: dict[str, Any],
    protocol: dict[str, Any],
    smoke: bool,
) -> tuple[GIBVAR, MinnesotaPosteriorVARGenerator, dict[str, float]]:
    gib_config = protocol["gibvar"]
    bvar_config = protocol["bvar"]
    n_gib = 4 if smoke else int(domain["gib_coefficient_systems"])
    n_bvar = 4 if smoke else int(bvar_config["posterior_draws"])
    burn_in = 8 if smoke else int(bvar_config["burn_in"])
    thin = 1 if smoke else int(bvar_config["thin"])
    timings: dict[str, float] = {}

    started = time.perf_counter()
    gib = GIBVAR(
        n_dynamics=n_gib,
        block_length=int(gib_config["block_length"]),
        expert_weight=float(gib_config["expert_weight"]),
        max_parents=int(gib_config["max_parents"]),
        max_spectral_radius=float(gib_config["stability_cap"]),
        random_state=stable_seed(7, "class8_gib_fit", str(Q0)),
        path_solver=str(domain["gib_path_solver"]),
        features=CLASS8_FEATURES,
        expert_edges=domain["expert_edges"],
        bounds=None,
    ).fit(history)
    timings["gibvar"] = time.perf_counter() - started

    started = time.perf_counter()
    bvar = MinnesotaPosteriorVARGenerator(
        n_dynamics=n_bvar,
        burn_in=burn_in,
        thin=thin,
        max_sampling_iterations=(
            2000 if smoke else int(bvar_config["maximum_sampling_iterations"])
        ),
        seed=stable_seed(int(bvar_config["fit_seed_base"]), "class8_bvar_fit", str(Q0)),
        features=CLASS8_FEATURES,
        bounds=domain["bounds"],
        own_lag_means=domain["bvar_own_lag_means"],
    ).fit(history)
    timings["minnesota_bvar"] = time.perf_counter() - started
    return gib, bvar, timings


def _sample_completions(
    gib: GIBVAR,
    bvar: MinnesotaPosteriorVARGenerator,
    history: pd.DataFrame,
    unemployment: np.ndarray,
    *,
    domain: dict[str, Any],
    protocol: dict[str, Any],
    smoke: bool,
    operational_bounds: Mapping[str, tuple[float, float]] | None = None,
    gaussian_only: bool = False,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, object],
]:
    student = protocol["student_t"]
    gib_config = protocol["gibvar"]
    n_paths = 48 if smoke else int(protocol["design"]["paths"])
    n_particles = 128 if smoke else int(student["conditional_particles"])
    state = history.iloc[-1].to_numpy(dtype=float)

    sampled_banks: dict[str, np.ndarray] = {}
    diagnostics: dict[str, object] = {}
    if not gaussian_only:
        gib_paths, gib_diagnostics = block_bridge_conditional_sample(
            gib,
            unemployment[:, None],
            ["unemployment"],
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
        sampled_banks[GIB_STUDENT_CASE] = np.asarray(gib_paths, dtype=float)
        diagnostics[GIB_STUDENT_CASE] = gib_diagnostics

    supplied = np.zeros((FUTURE_HORIZON, len(CLASS8_FEATURES)), dtype=float)
    unemployment_index = CLASS8_FEATURES.index("unemployment")
    supplied[:, unemployment_index] = unemployment
    gaussian_gib_paths, gaussian_gib_diagnostics = conditional_mixture_sample(
        gib,
        supplied,
        ["unemployment"],
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
        bvar_paths, bvar_diagnostics = conditional_mixture_sample(
            bvar,
            supplied,
            ["unemployment"],
            n_paths,
            np.random.default_rng(stable_seed(7, "class8_bvar_completion", str(Q0))),
            state,
            features=CLASS8_FEATURES,
            bounds=domain["bounds"],
            apply_hidden_bounds=False,
        )
        sampled_banks[BVAR_CASE] = np.asarray(bvar_paths, dtype=float)
        diagnostics[BVAR_CASE] = bvar_diagnostics

    effective_bounds = domain["bounds"] if operational_bounds is None else operational_bounds
    raw_banks, operational = _prepare_path_banks(
        sampled_banks,
        unemployment,
        n_paths=n_paths,
        operational_bounds=effective_bounds,
    )
    return raw_banks, operational, diagnostics


def _projection_metrics(
    projection: ClassProjection,
    *,
    method: str,
    path_id: str,
    bound_contact_fraction: float = np.nan,
) -> dict[str, float | int | str]:
    aggregate = projection.aggregate.sort_values("date")
    future_dates = set(aggregate["date"].iloc[1:].astype(int))
    detail = projection.detail.loc[projection.detail["date"].astype(int).isin(future_dates)]
    return {
        "method": method,
        "path_id": path_id,
        "minimum_cet1_ratio_pct": float(aggregate["cap_ratio"].min()),
        "end_cet1_ratio_pct": float(aggregate["cap_ratio"].iloc[-1]),
        "maximum_cet1_drawdown_bn": float(-aggregate["cet1_drop"].min()),
        # The public Y-9C archive stores dollar quantities in thousands; the
        # official CLASS output divides them by 1e6 to report USD billions.
        "cumulative_nco_bn": float(detail["qnetxoff_total"].sum() / 1_000_000.0),
        "cumulative_provisions_bn": float(detail["qprov"].sum() / 1_000_000.0),
        "cumulative_ppnr_bn": float(detail["ppnr"].sum() / 1_000_000.0),
        "cumulative_net_income_bn": float(detail["netinc"].sum() / 1_000_000.0),
        "firms_breaching": int(projection.breaches["breach"].sum()),
        "bound_contact_fraction": float(bound_contact_fraction),
    }


def _bound_contact_fraction(
    values: np.ndarray,
    domain: Mapping[str, Any],
    *,
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> float:
    contacts: list[np.ndarray] = []
    effective_bounds = domain["bounds"] if bounds is None else bounds
    for index, feature in enumerate(CLASS8_FEATURES):
        if feature == "unemployment":
            continue
        cells = values[:, index]
        contacts.append(_contact_mask(cells, effective_bounds[feature]))
    return float(np.concatenate(contacts).mean())


def _bound_contacts_by_feature(
    path_banks: Mapping[str, np.ndarray],
    bounds: Mapping[str, tuple[float, float]],
) -> pd.DataFrame:
    """Summarise operational-bound contacts by generator and feature."""
    rows: list[dict[str, float | str]] = []
    for method, paths in path_banks.items():
        for feature_index, feature in enumerate(CLASS8_FEATURES):
            contacts = _contact_mask(paths[:, :, feature_index], bounds[feature])
            within_path = contacts.mean(axis=1)
            rows.append(
                {
                    "method": method,
                    "feature": feature,
                    "cell_contact_fraction": float(contacts.mean()),
                    "share_paths_with_any_contact": float(contacts.any(axis=1).mean()),
                    "median_within_path_contact_fraction": float(
                        np.median(within_path)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _raw_path_diagnostics(
    path_banks: Mapping[str, np.ndarray],
    protocol_bounds: Mapping[str, list[float] | tuple[float, float]],
) -> pd.DataFrame:
    """Record raw ranges and violations of the original protocol bounds."""
    rows: list[dict[str, float | str]] = []
    for method, paths in path_banks.items():
        for feature_index, feature in enumerate(CLASS8_FEATURES):
            values = paths[:, :, feature_index].reshape(-1)
            lower, upper = (float(value) for value in protocol_bounds[feature])
            quantiles = np.quantile(values, [0.01, 0.05, 0.50, 0.95, 0.99])
            rows.append(
                {
                    "method": method,
                    "feature": feature,
                    "minimum": float(values.min()),
                    "p01": float(quantiles[0]),
                    "p05": float(quantiles[1]),
                    "median": float(quantiles[2]),
                    "p95": float(quantiles[3]),
                    "p99": float(quantiles[4]),
                    "maximum": float(values.max()),
                    "below_protocol_lower_fraction": float(np.mean(values < lower)),
                    "above_protocol_upper_fraction": float(np.mean(values > upper)),
                }
            )
    return pd.DataFrame(rows)


def _summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    values = (
        "minimum_cet1_ratio_pct",
        "end_cet1_ratio_pct",
        "maximum_cet1_drawdown_bn",
        "cumulative_nco_bn",
        "cumulative_provisions_bn",
        "cumulative_ppnr_bn",
        "cumulative_net_income_bn",
        "firms_breaching",
        "bound_contact_fraction",
    )
    rows: list[dict[str, float | int | str]] = []
    for method, group in metrics.groupby("method", sort=False):
        row: dict[str, float | int | str] = {
            "method": str(method),
            "n_paths": int(len(group)),
        }
        for column in values:
            data = group[column].to_numpy(dtype=float)
            for label, quantile in (("p05", 0.05), ("median", 0.50), ("p95", 0.95)):
                row[f"{column}_{label}"] = float(np.quantile(data, quantile))
        rows.append(row)
    return pd.DataFrame(rows)


def _write_results(
    output: Path,
    summary: pd.DataFrame,
    scenario_name: str,
    warning_summary: Mapping[str, Any],
    *,
    smoke: bool,
    treasury_floor_policy: str,
    generated_methods: tuple[str, ...] = GENERATED_METHODS,
) -> None:
    rate_policy_text = (
        "protocol Treasury lower bounds"
        if treasury_floor_policy == PROTOCOL_FLOOR_POLICY
        else "no lower bound on generated Treasury rates"
    )
    lines = [
        (
            "# Exploratory CLASS8-to-CLASS smoke comparison"
            if smoke
            else "# Exploratory CLASS8-to-CLASS full path-bank comparison"
        ),
        "",
        f"Condition: Minneapolis `{scenario_name}` unemployment path; 2019Q4 bank state;",
        "fixed DFAST market and operational-loss overlays excluded;",
        f"application rate policy: {rate_policy_text}.",
        "",
        "| Completion | Paths | Median minimum CET1 | Median cumulative NCO | "
        "Median cumulative PPNR | Median bound contacts |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        contacts = row.bound_contact_fraction_median
        contacts_text = "n/a" if not np.isfinite(contacts) else f"{100 * contacts:.1f}%"
        lines.append(
            f"| {row.method} | {row.n_paths} | "
            f"{row.minimum_cet1_ratio_pct_median:.2f}% | "
            f"${row.cumulative_nco_bn_median:.1f}bn | "
            f"${row.cumulative_ppnr_bn_median:.1f}bn | {contacts_text} |"
        )
    lines.extend(["", "Support warnings:", ""])
    for method in warning_summary:
        item = warning_summary[method]
        reasons = item["reasons"]
        text = "; ".join(reasons) if reasons else "none"
        lines.append(f"- {method}: {text}.")
    lines.extend(
        [
            "",
            (
                "This is a one-seed plumbing test using reduced fit and path counts."
                if smoke
                else "This is a one-seed, production-sized path-bank run."
            ),
            "It is not a report-final model ranking. Bound contacts",
            "indicate that completed paths still require support review before the CLASS",
            "outputs are interpreted substantively.",
            "The official row uses the archive's native 2019 seed history, whereas the",
            "generated rows use the revised Fed history through 2019Q4.",
        ]
    )
    if BVAR_CASE in generated_methods:
        lines.append(
            "Comparisons among the generated GIB--VAR and BVAR rows are fully matched."
        )
    else:
        lines.append(
            "The official row is therefore a contextual rather than fully matched comparator."
        )
    (output / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    output: Path,
    history_path: Path = DEFAULT_HISTORIC_CSV,
    protocol_path: Path = DEFAULT_PROTOCOL,
    archive: Path = DEFAULT_ARCHIVE,
    scenario_name: str = "fed_severe",
    smoke: bool = True,
    treasury_floor_policy: str = NO_LOWER_FLOOR_POLICY,
    gaussian_only: bool = False,
) -> Path:
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)

    protocol, domain = _load_domain(Path(protocol_path))
    selected_methods = _selected_methods(gaussian_only)
    operational_bounds = _operational_bounds(domain, treasury_floor_policy)
    model_history = load_us_class8_history(Path(history_path)).loc[:Q0].copy()
    if model_history.index.max() != Q0:
        raise ValueError("CLASS8 model history does not end at 2019Q4")
    actual_history = load_fed_class8_history(Path(history_path)).loc[:Q0].copy()
    official_macro, unemployment = _official_condition(
        Path(archive),
        scenario_name,
    )

    gib, bvar, fit_seconds = _fit_models(
        model_history,
        domain=domain,
        protocol=protocol,
        smoke=smoke,
    )
    raw_path_banks, path_banks, completion_diagnostics = _sample_completions(
        gib,
        bvar,
        model_history,
        unemployment,
        domain=domain,
        protocol=protocol,
        smoke=smoke,
        operational_bounds=operational_bounds,
        gaussian_only=gaussian_only,
    )
    unemployment_index = CLASS8_FEATURES.index("unemployment")
    restoration_errors = {
        method: {
            "raw": float(
                np.max(
                    np.abs(
                        raw_path_banks[method][:, :, unemployment_index]
                        - unemployment[None, :]
                    )
                )
            ),
            "operational": float(
                np.max(
                    np.abs(
                        path_banks[method][:, :, unemployment_index]
                        - unemployment[None, :]
                    )
                )
            ),
        }
        for method in selected_methods
    }
    if any(
        value != 0.0
        for method_errors in restoration_errors.values()
        for value in method_errors.values()
    ):
        raise RuntimeError("a supplied unemployment path was not restored exactly")
    np.savez_compressed(output / "raw_completed_paths.npz", **raw_path_banks)
    np.savez_compressed(output / "completed_paths.npz", **path_banks)
    bound_contacts = _bound_contacts_by_feature(path_banks, operational_bounds)
    raw_diagnostics = _raw_path_diagnostics(raw_path_banks, domain["bounds"])
    bound_contacts.to_csv(output / "bound_contacts_by_feature.csv", index=False)
    raw_diagnostics.to_csv(output / "raw_path_diagnostics.csv", index=False)

    warning_summary = _support_warning_summary(completion_diagnostics)
    support_payload = {
        "conditioned_features": ["unemployment"],
        "diagnostics": {
            method: _json_ready(value)
            for method, value in completion_diagnostics.items()
        },
        "warning_summary": warning_summary,
    }
    (output / "support_diagnostics.json").write_text(
        json.dumps(_json_ready(support_payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    projector = MinneapolisClassProjector.from_files(
        coefficients_path=Path(archive) / "output" / "estimated_model_coefficients.csv",
        y9c_path=Path(archive) / "y9c_bhc_data.csv",
        include_special_losses=False,
    )
    rows: list[dict[str, float | int | str]] = []
    official_projection = projector.project_path(official_macro)
    rows.append(
        _projection_metrics(
            official_projection,
            method=f"official_{scenario_name}",
            path_id=f"official_{scenario_name}",
        )
    )
    for method, paths in path_banks.items():
        print(f"projecting {len(paths)} {method} paths through CLASS", flush=True)
        for index, values in enumerate(paths):
            future = path_to_class8_frame(values, origin=Q0)
            macro = build_class_macro_input(
                actual_history,
                future,
                class_q0=Q0,
                scenario_name=f"{method}_{index:04d}",
            )
            projection = projector.project_path(macro)
            rows.append(
                _projection_metrics(
                    projection,
                    method=method,
                    path_id=f"{method}_{index:04d}",
                    bound_contact_fraction=_bound_contact_fraction(
                        values,
                        domain,
                        bounds=operational_bounds,
                    ),
                )
            )
            if (index + 1) % 100 == 0 or index + 1 == len(paths):
                print(
                    f"  {method}: projected {index + 1}/{len(paths)} paths",
                    flush=True,
                )

    metrics = pd.DataFrame(rows)
    summary = _summarize(metrics)
    metrics.to_csv(output / "path_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    _write_results(
        output,
        summary,
        scenario_name,
        warning_summary,
        smoke=smoke,
        treasury_floor_policy=treasury_floor_policy,
        generated_methods=selected_methods,
    )
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
        "study": "exploratory CLASS8-to-Minneapolis-CLASS pilot",
        "claim_eligible": False,
        "smoke": bool(smoke),
        "gaussian_only": bool(gaussian_only),
        "methods": list(selected_methods),
        "class_q0": str(Q0),
        "future_quarters": FUTURE_HORIZON,
        "reported_class_quarters": 9,
        "condition": f"Minneapolis {scenario_name} unemployment path",
        "path_policy": (
            "protocol-bounded operational copies with unemployment restored exactly"
            if treasury_floor_policy == PROTOCOL_FLOOR_POLICY
            else (
                "operational copies with no Treasury lower floor, all other "
                "protocol bounds retained, and unemployment restored exactly"
            )
        ),
        "treasury_rate_floor_policy": {
            "mode": treasury_floor_policy,
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
        "fit_seconds": (
            fit_seconds
            if not gaussian_only
            else {"gibvar": float(fit_seconds["gibvar"])}
        ),
        "sampling_seeds": {
            GIB_STUDENT_CASE: stable_seed(
                7, "class8_gib_completion", str(Q0)
            ),
            GIB_GAUSSIAN_CASE: GAUSSIAN_GIB_COMPLETION_SEED,
            BVAR_CASE: stable_seed(7, "class8_bvar_completion", str(Q0)),
        }
        if not gaussian_only
        else {GIB_GAUSSIAN_CASE: GAUSSIAN_GIB_COMPLETION_SEED},
        "n_paths": {name: int(len(paths)) for name, paths in path_banks.items()},
        "maximum_condition_restoration_error": restoration_errors,
        "diagnostic_types": {
            name: type(value).__name__ for name, value in completion_diagnostics.items()
        },
        "support_warning_summary": warning_summary,
        "path_artifacts": {
            "raw": "raw_completed_paths.npz",
            "operational": "completed_paths.npz",
        },
        "limitations": [
            (
                "The smoke run uses four fitted systems and 48 paths per generator."
                if smoke
                else (
                    "The run uses 25 GIB systems and 1,000 Gaussian GIB paths."
                    if gaussian_only
                    else "The run uses 25 GIB systems, 100 BVAR draws and 1,000 paths per generator."
                )
            ),
            "The bank state and CLASS coefficients are fixed at 2019Q4.",
            "Satellite parameter uncertainty is not propagated.",
            "The official row uses archive-native seed history and is not directly matched to the generated rows.",
            "The comparison is exploratory and not report-final evidence.",
        ],
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORIC_CSV)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument(
        "--condition-scenario",
        default="fed_severe",
        help="Scenario in the Minneapolis macro file whose unemployment path is supplied.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the complete protocol sizes instead of the quick diagnostic sizes.",
    )
    parser.add_argument(
        "--treasury-floor-policy",
        choices=TREASURY_FLOOR_POLICIES,
        default=NO_LOWER_FLOOR_POLICY,
        help=(
            "Use protocol Treasury lower bounds or remove only their lower "
            "floor before CLASS projection."
        ),
    )
    parser.add_argument(
        "--gaussian-only",
        action="store_true",
        help="Generate and project only Gaussian GIB paths plus the official comparator.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    output = run(
        output=args.out,
        history_path=args.history,
        protocol_path=args.protocol,
        archive=args.archive,
        scenario_name=str(args.condition_scenario),
        smoke=not bool(args.full),
        treasury_floor_policy=str(args.treasury_floor_policy),
        gaussian_only=bool(args.gaussian_only),
    )
    print(f"wrote CLASS generator comparison: {output}", flush=True)


if __name__ == "__main__":
    main()
