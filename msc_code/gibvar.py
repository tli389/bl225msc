"""Standalone Graph-Informed Bootstrap VAR with consecutive shock blocks.

This file isolates the maintained thesis implementation of GIB-VAR from the
legacy SSP variants in the research codebase.  The important version choice is
deliberate:

* coefficient systems are refitted from moving blocks of consecutive
  transition pairs;
* every refitted system's residual pool is recomputed against the original
  chronological training history; and
* future innovations are sampled as non-circular blocks of consecutive
  multivariate residual rows.

The older ``bootstrap_order`` residual pool is intentionally not supported.
That pool could join residual rows that were adjacent only because transition
blocks had been pasted together during the coefficient bootstrap.  Here,
within-block adjacency always means adjacency in the observed history.

The implementation is self-contained apart from NumPy, pandas, and
scikit-learn.  The historical seven-variable feature map remains the class
default for backwards compatibility; the report-aligned CLASS8 and UK8
runners pass their feature lists and preferred maps explicitly.  The shared
design defaults are 25 coefficient systems, four-quarter blocks, graph penalty
weight 0.35, at most three cross-variable parents per equation, and a VAR
stability cap of 0.995.

Example
-------
>>> model = GIBVAR().fit(history)
>>> paths = model.sample(
...     n_paths=1000,
...     horizon=12,
...     rng=np.random.default_rng(7),
... )
>>> paths.shape
(1000, 12, 7)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, lars_path, lasso_path
from sklearn.model_selection import TimeSeriesSplit

DEFAULT_FEATURES: tuple[str, ...] = (
    "gdp_growth",
    "fed_funds",
    "mortgage_30y",
    "bbb_spread",
    "unemployment",
    "hpi_qoq_growth",
    "vix",
)

# Parent -> target preferences are represented as target: [parents].  These
# edges lower the weighted-Lasso penalty; they never force an edge, sign, or
# coefficient magnitude into the fitted system.
DEFAULT_EXPERT_EDGES: dict[str, tuple[str, ...]] = {
    "gdp_growth": ("fed_funds",),
    "fed_funds": ("unemployment",),
    "mortgage_30y": ("fed_funds",),
    "bbb_spread": ("vix",),
    "unemployment": ("gdp_growth", "bbb_spread"),
    "hpi_qoq_growth": ("unemployment", "mortgage_30y"),
    "vix": ("bbb_spread",),
}

DEFAULT_BOUNDS: dict[str, tuple[float, float]] = {
    "gdp_growth": (-15.0, 12.0),
    "fed_funds": (0.0, 20.0),
    "mortgage_30y": (0.0, 20.0),
    "bbb_spread": (0.0, 12.0),
    "unemployment": (0.0, 25.0),
    "hpi_qoq_growth": (-15.0, 10.0),
    "vix": (5.0, 120.0),
}

DEFAULT_COEFFICIENT_DRAWS = 25
DEFAULT_BLOCK_LENGTH = 4
DEFAULT_EXPERT_WEIGHT = 0.35
DEFAULT_MAX_PARENTS = 3
DEFAULT_MAX_SPECTRAL_RADIUS = 0.995


@dataclass
class FittedDynamics:
    """One fitted stable VAR(1) coefficient system and its shock pool."""

    coef: np.ndarray
    intercept: np.ndarray
    residuals: np.ndarray
    residual_cov: np.ndarray
    raw_spectral_radius: float
    stability_scale: float
    # Total first-moment shift absorbed into ``intercept`` relative to the
    # uncentred fitted representation.  Bootstrap draws can be centred once
    # on their resampled fit and again after their residual pool is rebuilt on
    # the chronological history, so this field is deliberately cumulative.
    residual_centering_shift: np.ndarray


@dataclass
class _PartialledDesign:
    """Fold-fitted transformation for one weighted-Lasso equation."""

    target: int
    cross: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_projection: np.ndarray
    x_projection: np.ndarray
    residual_scale: np.ndarray
    penalty_weights: np.ndarray
    design: np.ndarray
    target_residual: np.ndarray


@dataclass
class _HardGraphDesign:
    """Fold-fitted transformation with graph parents left unpenalised."""

    unpenalized: np.ndarray
    optional: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_projection: np.ndarray
    x_projection: np.ndarray
    residual_scale: np.ndarray
    design: np.ndarray
    target_residual: np.ndarray


def moving_block_pair_indices(
    n_pairs: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample transition-pair indices in non-circular consecutive blocks.

    Independent blocks may meet at an artificial boundary, but every row
    *inside* a block follows the next row in the original chronology.  The
    returned vector is truncated to exactly ``n_pairs`` rows.
    """

    if n_pairs < 1:
        raise ValueError("n_pairs must be positive")
    if block_length < 1:
        raise ValueError("block_length must be positive")
    length = min(int(block_length), n_pairs)
    n_blocks = int(np.ceil(n_pairs / length))
    starts = rng.integers(0, n_pairs - length + 1, size=n_blocks)
    indices = np.concatenate([start + np.arange(length) for start in starts])
    return indices[:n_pairs]


def consecutive_block_indices(
    n_rows: int,
    n_paths: int,
    horizon: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw non-circular consecutive residual indices for future paths.

    Each output row has shape ``(horizon,)``.  For a 12-quarter horizon and a
    four-quarter block length, positions 0--3, 4--7, and 8--11 are each
    consecutive sections of the historical residual pool.  A new independent
    start is drawn at each block boundary.
    """

    if n_rows < 1:
        raise ValueError("n_rows must be positive")
    if n_paths < 1:
        raise ValueError("n_paths must be positive")
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if block_length < 1:
        raise ValueError("block_length must be positive")

    length = min(int(block_length), n_rows, horizon)
    n_blocks = int(np.ceil(horizon / length))
    starts = rng.integers(
        0,
        n_rows - length + 1,
        size=(n_paths, n_blocks),
    )
    offsets = np.arange(length)
    indices = (starts[:, :, None] + offsets[None, None, :]).reshape(
        n_paths,
        n_blocks * length,
    )
    return indices[:, :horizon]


def _regularized_covariance(residuals: np.ndarray) -> np.ndarray:
    covariance = np.atleast_2d(np.cov(residuals, rowvar=False))
    diagonal = np.maximum(np.diag(covariance), 1e-6)
    return covariance + np.diag(0.05 * diagonal + 1e-8)


def _center_residual_law(
    intercept: np.ndarray,
    residuals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Move an empirical residual mean into the dynamics intercept.

    For every stored innovation row ``e_t``, the identity
    ``c + e_t == (c + mean(e)) + (e_t - mean(e))`` holds.  Native block
    paths and smoothed-block paths are therefore unchanged, while Gaussian
    approximations now share the same first-moment law instead of silently
    treating an uncentred residual pool as zero mean.
    """

    values = np.asarray(residuals, dtype=float)
    if values.ndim != 2 or not len(values):
        raise ValueError("residuals must be a non-empty two-dimensional array")
    shift = values.mean(axis=0)
    centered = values - shift[None, :]
    adjusted_intercept = np.asarray(intercept, dtype=float) + shift
    return adjusted_intercept, centered, shift


def _stabilize_lag_matrix(
    coef: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    max_radius: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Project a VAR(1) into the stable region without changing its support."""

    raw_radius = float(np.max(np.abs(np.linalg.eigvals(coef))))
    scale = 1.0 if raw_radius <= max_radius else max_radius / raw_radius
    stable_coef = coef * scale
    intercept = y.mean(axis=0) - x.mean(axis=0) @ stable_coef.T
    return stable_coef, intercept, raw_radius, float(scale)


def _time_series_cv(n_rows: int) -> TimeSeriesSplit:
    if n_rows < 12:
        raise ValueError("at least 12 transition rows are required")
    n_splits = min(5, max(2, n_rows // 12))
    return TimeSeriesSplit(n_splits=n_splits)


def _partialled_design(
    x: np.ndarray,
    target_values: np.ndarray,
    target: int,
    penalty_weights: np.ndarray,
) -> _PartialledDesign:
    """Partial out the intercept and unpenalised own lag."""

    n_rows, n_features = x.shape
    cross = np.asarray(
        [index for index in range(n_features) if index != target],
        dtype=int,
    )
    x_mean = x.mean(axis=0)
    x_scale = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
    standardized = (x - x_mean) / x_scale
    own_design = np.column_stack(
        [np.ones(n_rows), standardized[:, target]],
    )

    y_projection, *_ = np.linalg.lstsq(
        own_design,
        target_values,
        rcond=None,
    )
    target_residual = target_values - own_design @ y_projection
    cross_values = standardized[:, cross]
    x_projection, *_ = np.linalg.lstsq(
        own_design,
        cross_values,
        rcond=None,
    )
    x_residual = cross_values - own_design @ x_projection
    residual_scale = np.where(
        x_residual.std(axis=0) > 1e-8,
        x_residual.std(axis=0),
        1.0,
    )
    design = x_residual / residual_scale / penalty_weights[None, :]
    return _PartialledDesign(
        target=target,
        cross=cross,
        x_mean=x_mean,
        x_scale=x_scale,
        y_projection=y_projection,
        x_projection=x_projection,
        residual_scale=residual_scale,
        penalty_weights=penalty_weights,
        design=design,
        target_residual=target_residual,
    )


def _transform_partialled(
    fitted: _PartialledDesign,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a training-fold transformation to validation rows."""

    standardized = (x - fitted.x_mean) / fitted.x_scale
    own_design = np.column_stack(
        [np.ones(len(x)), standardized[:, fitted.target]],
    )
    baseline = own_design @ fitted.y_projection
    cross_values = standardized[:, fitted.cross]
    residual = cross_values - own_design @ fitted.x_projection
    design = residual / fitted.residual_scale / fitted.penalty_weights[None, :]
    return design, baseline


def _alpha_ratio_fold_local(
    x: np.ndarray,
    target_values: np.ndarray,
    target: int,
    penalty_weights: np.ndarray,
    path_solver: str,
) -> float:
    """Select a scale-free penalty using forward, fold-local validation."""

    if path_solver not in {"coordinate", "lars"}:
        raise ValueError("path_solver must be 'coordinate' or 'lars'")
    ratios = np.geomspace(1.0, 1e-3, 60)
    losses = np.zeros(len(ratios), dtype=float)
    folds = list(_time_series_cv(len(x)).split(x))

    for train_index, validation_index in folds:
        fitted = _partialled_design(
            x[train_index],
            target_values[train_index],
            target,
            penalty_weights,
        )
        validation_design, baseline = _transform_partialled(
            fitted,
            x[validation_index],
        )
        alpha_max = float(
            np.max(np.abs(fitted.design.T @ fitted.target_residual))
            / max(len(train_index), 1)
        )
        if alpha_max <= 1e-12 or np.std(fitted.target_residual) <= 1e-10:
            predictions = np.repeat(baseline[:, None], len(ratios), axis=1)
        else:
            alphas = np.maximum(alpha_max * ratios, 1e-12)
            if path_solver == "coordinate":
                _, coefficients, _ = lasso_path(
                    fitted.design,
                    fitted.target_residual,
                    alphas=alphas,
                    max_iter=100_000,
                    tol=5e-4,
                )
            else:
                path_alphas, _, path_coefficients = lars_path(
                    fitted.design,
                    fitted.target_residual,
                    method="lasso",
                    alpha_min=0.0,
                )
                order = np.argsort(path_alphas)
                unique_alphas, unique_indices = np.unique(
                    path_alphas[order],
                    return_index=True,
                )
                ordered = path_coefficients[:, order][:, unique_indices]
                coefficients = np.vstack(
                    [
                        np.interp(alphas, unique_alphas, coefficient_path)
                        for coefficient_path in ordered
                    ]
                )
            predictions = baseline[:, None] + validation_design @ coefficients
        errors = target_values[validation_index, None] - predictions
        losses += np.mean(errors**2, axis=0)

    return float(ratios[int(np.argmin(losses / len(folds)))])


def _fit_sparse_target(
    x: np.ndarray,
    target_values: np.ndarray,
    target: int,
    max_parents: int,
    penalty_weights: np.ndarray,
    path_solver: str,
) -> tuple[np.ndarray, float]:
    """Fit one weighted-Lasso equation with an unpenalised own lag."""

    n_rows, n_features = x.shape
    cross = np.asarray(
        [index for index in range(n_features) if index != target],
        dtype=int,
    )
    penalty_weights = np.asarray(penalty_weights, dtype=float)
    if penalty_weights.shape != (len(cross),) or np.any(penalty_weights <= 0):
        raise ValueError("penalty weights must be positive and match cross-lags")

    ratio = _alpha_ratio_fold_local(
        x,
        target_values,
        target,
        penalty_weights,
        path_solver,
    )
    fitted = _partialled_design(x, target_values, target, penalty_weights)
    beta_weighted = np.zeros(len(cross), dtype=float)

    if np.std(fitted.target_residual) > 1e-10:
        alpha_max = float(
            np.max(np.abs(fitted.design.T @ fitted.target_residual)) / max(n_rows, 1)
        )
        selector = Lasso(
            alpha=max(alpha_max * ratio, 1e-12),
            fit_intercept=False,
            max_iter=100_000,
        ).fit(fitted.design, fitted.target_residual)
        ranked = np.argsort(np.abs(selector.coef_))[::-1]
        selected = [
            int(index) for index in ranked if abs(float(selector.coef_[index])) > 1e-12
        ][:max_parents]
        if selected:
            refit = Lasso(
                alpha=float(selector.alpha),
                fit_intercept=False,
                max_iter=100_000,
            ).fit(fitted.design[:, selected], fitted.target_residual)
            beta_weighted[selected] = np.asarray(refit.coef_, dtype=float)

    beta_cross_z = beta_weighted / fitted.penalty_weights / fitted.residual_scale
    standardized = (x - fitted.x_mean) / fitted.x_scale
    cross_values = standardized[:, cross]
    own_design = np.column_stack(
        [np.ones(n_rows), standardized[:, target]],
    )
    partial_target = target_values - cross_values @ beta_cross_z
    own_coef, *_ = np.linalg.lstsq(
        own_design,
        partial_target,
        rcond=None,
    )
    beta_z = np.zeros(n_features, dtype=float)
    beta_z[cross] = beta_cross_z
    beta_z[target] = float(own_coef[1])
    beta = beta_z / fitted.x_scale
    intercept = float(own_coef[0]) - float(beta @ fitted.x_mean)
    return beta, intercept


def _hard_graph_design(
    x: np.ndarray,
    target_values: np.ndarray,
    target: int,
    forced_parents: np.ndarray,
) -> _HardGraphDesign:
    """Partial out the intercept, own lag, and required graph parents."""

    n_rows, n_features = x.shape
    forced = np.asarray(forced_parents, dtype=int)
    unpenalized = np.concatenate(([target], forced))
    optional = np.asarray(
        [
            index
            for index in range(n_features)
            if index != target and index not in set(forced.tolist())
        ],
        dtype=int,
    )
    x_mean = x.mean(axis=0)
    x_scale = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
    standardized = (x - x_mean) / x_scale
    base_design = np.column_stack(
        [np.ones(n_rows), standardized[:, unpenalized]],
    )
    y_projection, *_ = np.linalg.lstsq(
        base_design,
        target_values,
        rcond=None,
    )
    target_residual = target_values - base_design @ y_projection

    if len(optional):
        optional_values = standardized[:, optional]
        x_projection, *_ = np.linalg.lstsq(
            base_design,
            optional_values,
            rcond=None,
        )
        x_residual = optional_values - base_design @ x_projection
        residual_scale = np.where(
            x_residual.std(axis=0) > 1e-8,
            x_residual.std(axis=0),
            1.0,
        )
        design = x_residual / residual_scale
    else:
        x_projection = np.empty((base_design.shape[1], 0), dtype=float)
        residual_scale = np.empty(0, dtype=float)
        design = np.empty((n_rows, 0), dtype=float)

    return _HardGraphDesign(
        unpenalized=unpenalized,
        optional=optional,
        x_mean=x_mean,
        x_scale=x_scale,
        y_projection=y_projection,
        x_projection=x_projection,
        residual_scale=residual_scale,
        design=design,
        target_residual=target_residual,
    )


def _transform_hard_graph_design(
    fitted: _HardGraphDesign,
    x: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a hard-graph training-fold transformation to validation rows."""

    standardized = (x - fitted.x_mean) / fitted.x_scale
    base_design = np.column_stack(
        [np.ones(len(x)), standardized[:, fitted.unpenalized]],
    )
    baseline = base_design @ fitted.y_projection
    if not len(fitted.optional):
        return np.empty((len(x), 0), dtype=float), baseline
    optional_values = standardized[:, fitted.optional]
    residual = optional_values - base_design @ fitted.x_projection
    return residual / fitted.residual_scale, baseline


def _hard_graph_alpha_ratio_fold_local(
    x: np.ndarray,
    target_values: np.ndarray,
    target: int,
    forced_parents: np.ndarray,
    path_solver: str,
) -> float:
    """Select the optional-edge penalty with forward validation."""

    ratios = np.geomspace(1.0, 1e-3, 60)
    losses = np.zeros(len(ratios), dtype=float)
    folds = list(_time_series_cv(len(x)).split(x))

    for train_index, validation_index in folds:
        fitted = _hard_graph_design(
            x[train_index],
            target_values[train_index],
            target,
            forced_parents,
        )
        validation_design, baseline = _transform_hard_graph_design(
            fitted,
            x[validation_index],
        )
        alpha_max = float(
            np.max(np.abs(fitted.design.T @ fitted.target_residual))
            / max(len(train_index), 1)
        )
        if alpha_max <= 1e-12 or np.std(fitted.target_residual) <= 1e-10:
            predictions = np.repeat(baseline[:, None], len(ratios), axis=1)
        else:
            alphas = np.maximum(alpha_max * ratios, 1e-12)
            if path_solver == "coordinate":
                _, coefficients, _ = lasso_path(
                    fitted.design,
                    fitted.target_residual,
                    alphas=alphas,
                    max_iter=100_000,
                    tol=5e-4,
                )
            else:
                path_alphas, _, path_coefficients = lars_path(
                    fitted.design,
                    fitted.target_residual,
                    method="lasso",
                    alpha_min=0.0,
                )
                order = np.argsort(path_alphas)
                unique_alphas, unique_indices = np.unique(
                    path_alphas[order],
                    return_index=True,
                )
                ordered = path_coefficients[:, order][:, unique_indices]
                coefficients = np.vstack(
                    [
                        np.interp(alphas, unique_alphas, coefficient_path)
                        for coefficient_path in ordered
                    ]
                )
            predictions = baseline[:, None] + validation_design @ coefficients
        errors = target_values[validation_index, None] - predictions
        losses += np.mean(errors**2, axis=0)

    return float(ratios[int(np.argmin(losses / len(folds)))])


def _fit_hard_graph_target(
    x: np.ndarray,
    target_values: np.ndarray,
    target: int,
    max_parents: int,
    forced_parents: np.ndarray,
    path_solver: str,
) -> tuple[np.ndarray, float]:
    """Fit one equation with graph parents required but data-estimated."""

    n_rows, n_features = x.shape
    forced = np.unique(np.asarray(forced_parents, dtype=int))
    if target in forced:
        raise ValueError("the own lag cannot also be a forced graph parent")
    if len(forced) > max_parents:
        raise ValueError("forced graph parents exceed the cross-parent cap")

    fitted = _hard_graph_design(x, target_values, target, forced)
    beta_optional_z = np.zeros(len(fitted.optional), dtype=float)
    optional_slots = max_parents - len(forced)

    if (
        optional_slots > 0
        and len(fitted.optional)
        and np.std(fitted.target_residual) > 1e-10
    ):
        ratio = _hard_graph_alpha_ratio_fold_local(
            x,
            target_values,
            target,
            forced,
            path_solver,
        )
        alpha_max = float(
            np.max(np.abs(fitted.design.T @ fitted.target_residual))
            / max(n_rows, 1)
        )
        selector = Lasso(
            alpha=max(alpha_max * ratio, 1e-12),
            fit_intercept=False,
            max_iter=100_000,
        ).fit(fitted.design, fitted.target_residual)
        ranked = np.argsort(np.abs(selector.coef_))[::-1]
        selected = [
            int(index)
            for index in ranked
            if abs(float(selector.coef_[index])) > 1e-12
        ][:optional_slots]
        if selected:
            refit = Lasso(
                alpha=float(selector.alpha),
                fit_intercept=False,
                max_iter=100_000,
            ).fit(fitted.design[:, selected], fitted.target_residual)
            beta_optional_z[selected] = (
                np.asarray(refit.coef_, dtype=float)
                / fitted.residual_scale[selected]
            )

    standardized = (x - fitted.x_mean) / fitted.x_scale
    partial_target = target_values.copy()
    if len(fitted.optional):
        partial_target -= standardized[:, fitted.optional] @ beta_optional_z
    base_design = np.column_stack(
        [np.ones(n_rows), standardized[:, fitted.unpenalized]],
    )
    base_coef, *_ = np.linalg.lstsq(
        base_design,
        partial_target,
        rcond=None,
    )

    beta_z = np.zeros(n_features, dtype=float)
    beta_z[fitted.optional] = beta_optional_z
    beta_z[fitted.unpenalized] = base_coef[1:]
    beta = beta_z / fitted.x_scale
    intercept = float(base_coef[0]) - float(beta @ fitted.x_mean)
    return beta, intercept


def _fit_hard_graph_dynamics(
    x: np.ndarray,
    y: np.ndarray,
    *,
    features: Sequence[str],
    expert_edges: Mapping[str, Sequence[str]],
    max_parents: int,
    max_spectral_radius: float,
    path_solver: str,
) -> FittedDynamics:
    """Fit a sparse VAR with preferred graph parents required in each equation."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape != y.shape:
        raise ValueError("x and y must be equal-shape two-dimensional arrays")
    if len(x) < 12:
        raise ValueError("at least 12 transition rows are required")
    if x.shape[1] != len(features):
        raise ValueError("array width does not match the feature list")

    n_features = x.shape[1]
    coef = np.zeros((n_features, n_features), dtype=float)
    intercept = np.zeros(n_features, dtype=float)
    for target in range(n_features):
        preferred = set(expert_edges.get(features[target], ()))
        forced = np.asarray(
            [
                index
                for index in range(n_features)
                if index != target and features[index] in preferred
            ],
            dtype=int,
        )
        coef[target], intercept[target] = _fit_hard_graph_target(
            x,
            y[:, target],
            target,
            max_parents,
            forced,
            path_solver,
        )

    coef, intercept, raw_radius, stability_scale = _stabilize_lag_matrix(
        coef,
        x,
        y,
        max_spectral_radius,
    )
    residuals = y - (x @ coef.T + intercept[None, :])
    residual_covariance = _regularized_covariance(residuals)
    intercept, residuals, centering_shift = _center_residual_law(
        intercept,
        residuals,
    )
    return FittedDynamics(
        coef=coef,
        intercept=intercept,
        residuals=residuals,
        residual_cov=residual_covariance,
        raw_spectral_radius=raw_radius,
        stability_scale=stability_scale,
        residual_centering_shift=centering_shift,
    )


def _fit_graph_informed_dynamics(
    x: np.ndarray,
    y: np.ndarray,
    *,
    features: Sequence[str],
    expert_edges: Mapping[str, Sequence[str]],
    expert_weight: float,
    max_parents: int,
    max_spectral_radius: float,
    path_solver: str,
) -> FittedDynamics:
    """Fit the graph-weighted sparse VAR(1) to aligned transition arrays."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or x.shape != y.shape:
        raise ValueError("x and y must be equal-shape two-dimensional arrays")
    if len(x) < 12:
        raise ValueError("at least 12 transition rows are required")
    if x.shape[1] != len(features):
        raise ValueError("array width does not match the feature list")

    n_features = x.shape[1]
    coef = np.zeros((n_features, n_features), dtype=float)
    intercept = np.zeros(n_features, dtype=float)

    for target in range(n_features):
        cross = np.asarray(
            [index for index in range(n_features) if index != target],
            dtype=int,
        )
        preferred_parents = set(expert_edges.get(features[target], ()))
        penalty_weights = np.asarray(
            [
                expert_weight if features[index] in preferred_parents else 1.0
                for index in cross
            ],
            dtype=float,
        )
        coef[target], intercept[target] = _fit_sparse_target(
            x,
            y[:, target],
            target,
            max_parents,
            penalty_weights,
            path_solver,
        )

    coef, intercept, raw_radius, stability_scale = _stabilize_lag_matrix(
        coef,
        x,
        y,
        max_spectral_radius,
    )
    residuals = y - (x @ coef.T + intercept[None, :])
    # Compute the covariance from the original pool.  Covariance is
    # translation-invariant, but preserving this ordering also preserves the
    # pre-centring floating-point covariance used by the Gaussian and bridge
    # kernels.
    residual_covariance = _regularized_covariance(residuals)
    intercept, residuals, centering_shift = _center_residual_law(
        intercept,
        residuals,
    )
    return FittedDynamics(
        coef=coef,
        intercept=intercept,
        residuals=residuals,
        residual_cov=residual_covariance,
        raw_spectral_radius=raw_radius,
        stability_scale=stability_scale,
        residual_centering_shift=centering_shift,
    )


class GIBVAR:
    """Graph-Informed Bootstrap VAR using chronological residual blocks.

    ``fit`` estimates one reference system and a bootstrap mixture of
    coefficient systems.  ``sample`` assigns one fixed coefficient system to
    each path and recursively rolls it forward with consecutive historical
    multivariate shock blocks.
    """

    name = "gibvar_chronological_blocks"

    def __init__(
        self,
        n_dynamics: int = DEFAULT_COEFFICIENT_DRAWS,
        block_length: int = DEFAULT_BLOCK_LENGTH,
        expert_weight: float = DEFAULT_EXPERT_WEIGHT,
        max_parents: int = DEFAULT_MAX_PARENTS,
        max_spectral_radius: float = DEFAULT_MAX_SPECTRAL_RADIUS,
        random_state: int = 10_000,
        path_solver: str = "coordinate",
        *,
        features: Sequence[str] = DEFAULT_FEATURES,
        expert_edges: Mapping[str, Sequence[str]] = DEFAULT_EXPERT_EDGES,
        bounds: Mapping[str, tuple[float, float]] | None = DEFAULT_BOUNDS,
        force_expert_edges: bool = False,
    ) -> None:
        if n_dynamics < 1:
            raise ValueError("n_dynamics must be positive")
        if block_length < 1:
            raise ValueError("block_length must be positive")
        if not 0.0 < expert_weight <= 1.0:
            raise ValueError("expert_weight must lie in (0, 1]")
        if max_parents < 1:
            raise ValueError("max_parents must be positive")
        if not 0.0 < max_spectral_radius < 1.0:
            raise ValueError("max_spectral_radius must lie in (0, 1)")
        if path_solver not in {"coordinate", "lars"}:
            raise ValueError("path_solver must be 'coordinate' or 'lars'")

        feature_names = tuple(features)
        if not feature_names or len(set(feature_names)) != len(feature_names):
            raise ValueError("features must be non-empty and unique")

        self.n_dynamics = int(n_dynamics)
        self.block_length = int(block_length)
        self.expert_weight = float(expert_weight)
        self.max_parents = int(max_parents)
        self.max_spectral_radius = float(max_spectral_radius)
        self.random_state = int(random_state)
        self.path_solver = path_solver
        self.features = feature_names
        self.expert_edges = {
            target: tuple(parents) for target, parents in expert_edges.items()
        }
        self.bounds = None if bounds is None else dict(bounds)
        self.force_expert_edges = bool(force_expert_edges)

        self.dynamics_: FittedDynamics | None = None
        self.draws_: list[FittedDynamics] | None = None
        self.bootstrap_indices_: list[np.ndarray] | None = None
        self.jumpoff_: np.ndarray | None = None
        self.last_draw_assignments_: np.ndarray | None = None
        self.last_shock_indices_: np.ndarray | None = None

    def _validated_history(self, history: pd.DataFrame) -> np.ndarray:
        if not isinstance(history, pd.DataFrame):
            raise TypeError("history must be a pandas DataFrame")
        missing = [name for name in self.features if name not in history.columns]
        if missing:
            raise ValueError(f"history is missing features: {missing}")
        values = history.loc[:, self.features].to_numpy(dtype=float)
        if len(values) < 13:
            raise ValueError("history must contain at least 13 observations")
        if not np.isfinite(values).all():
            raise ValueError("history contains non-finite values")
        return values

    def fit(self, history: pd.DataFrame) -> GIBVAR:
        """Fit the point system and bootstrap coefficient mixture.

        The bootstrap determines coefficient estimates only.  After every
        refit, residuals are recomputed on ``values[:-1] -> values[1:]`` in
        their original chronological order.  This is the maintained version
        of GIB-VAR and the key correction relative to the older code.
        """

        values = self._validated_history(history)
        chronological_x = values[:-1]
        chronological_y = values[1:]
        if self.force_expert_edges:
            fit_function = _fit_hard_graph_dynamics
            fit_arguments = {
                "features": self.features,
                "expert_edges": self.expert_edges,
                "max_parents": self.max_parents,
                "max_spectral_radius": self.max_spectral_radius,
                "path_solver": self.path_solver,
            }
        else:
            fit_function = _fit_graph_informed_dynamics
            fit_arguments = {
                "features": self.features,
                "expert_edges": self.expert_edges,
                "expert_weight": self.expert_weight,
                "max_parents": self.max_parents,
                "max_spectral_radius": self.max_spectral_radius,
                "path_solver": self.path_solver,
            }

        self.dynamics_ = fit_function(
            chronological_x,
            chronological_y,
            **fit_arguments,
        )

        rng = np.random.default_rng(self.random_state + len(values))
        draws: list[FittedDynamics] = []
        bootstrap_indices: list[np.ndarray] = []
        n_pairs = len(chronological_x)

        for _ in range(self.n_dynamics):
            chosen = moving_block_pair_indices(
                n_pairs,
                self.block_length,
                rng,
            )
            candidate = fit_function(
                values[chosen],
                values[chosen + 1],
                **fit_arguments,
            )

            # Maintained chronological pool: do not retain candidate.residuals,
            # because those rows follow the pasted bootstrap-block order.
            residuals = chronological_y - (
                chronological_x @ candidate.coef.T + candidate.intercept[None, :]
            )
            residual_covariance = _regularized_covariance(residuals)
            intercept, residuals, centering_shift = _center_residual_law(
                candidate.intercept,
                residuals,
            )
            cumulative_centering_shift = (
                candidate.residual_centering_shift + centering_shift
            )
            draws.append(
                FittedDynamics(
                    coef=candidate.coef,
                    intercept=intercept,
                    residuals=residuals,
                    residual_cov=residual_covariance,
                    raw_spectral_radius=candidate.raw_spectral_radius,
                    stability_scale=candidate.stability_scale,
                    residual_centering_shift=cumulative_centering_shift,
                )
            )
            bootstrap_indices.append(chosen.copy())

        self.draws_ = draws
        self.bootstrap_indices_ = bootstrap_indices
        self.jumpoff_ = values[-1].copy()
        self.last_draw_assignments_ = None
        self.last_shock_indices_ = None
        return self

    def _require_fitted(self) -> list[FittedDynamics]:
        if self.draws_ is None or self.dynamics_ is None or self.jumpoff_ is None:
            raise RuntimeError("GIBVAR must be fitted before sampling")
        return self.draws_

    def draw_innovations(
        self,
        n_paths: int,
        horizon: int,
        rng: np.random.Generator | None = None,
    ) -> np.ndarray:
        """Draw consecutive-block innovations from the coefficient mixture."""

        draws = self._require_fitted()
        if n_paths < 1 or horizon < 1:
            raise ValueError("n_paths and horizon must be positive")
        if rng is None:
            rng = np.random.default_rng()

        assignments = rng.integers(0, len(draws), size=n_paths)
        innovations = np.empty((n_paths, horizon, len(self.features)))
        shock_indices = np.empty((n_paths, horizon), dtype=int)

        for draw_index in np.unique(assignments):
            mask = assignments == draw_index
            pool = draws[int(draw_index)].residuals
            indices = consecutive_block_indices(
                len(pool),
                int(mask.sum()),
                horizon,
                self.block_length,
                rng,
            )
            innovations[mask] = pool[indices]
            shock_indices[mask] = indices

        self.last_draw_assignments_ = assignments.copy()
        self.last_shock_indices_ = shock_indices.copy()
        return innovations

    def sample(
        self,
        n_paths: int,
        horizon: int,
        rng: np.random.Generator | None = None,
        jumpoff: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate an ensemble with one fixed coefficient draw per path.

        Without ``anchor``, paths are recursive levels beginning at ``jumpoff``;
        the final fitted observation is used when ``jumpoff`` is omitted.
        With ``anchor`` (shape ``(horizon, n_features)``), GIB-VAR propagates
        stochastic deviations around the supplied deterministic path.
        """

        draws = self._require_fitted()
        innovations = self.draw_innovations(n_paths, horizon, rng)
        assignments = self.last_draw_assignments_
        n_features = len(self.features)
        paths = np.empty((n_paths, horizon, n_features), dtype=float)

        if anchor is not None:
            anchor_values = np.asarray(anchor, dtype=float)
            if anchor_values.shape != (horizon, n_features):
                raise ValueError("anchor must have shape (horizon, number of features)")
        else:
            anchor_values = None

        initial_state = (
            self.jumpoff_
            if jumpoff is None
            else np.asarray(
                jumpoff,
                dtype=float,
            )
        )
        if initial_state.shape != (n_features,):
            raise ValueError("jumpoff must have one value per feature")

        for draw_index in np.unique(assignments):
            mask = assignments == draw_index
            dynamics = draws[int(draw_index)]
            group_size = int(mask.sum())

            if anchor_values is not None:
                deviation = np.zeros((group_size, n_features), dtype=float)
                for step in range(horizon):
                    deviation = deviation @ dynamics.coef.T + innovations[mask, step]
                    paths[mask, step] = anchor_values[step] + deviation
            else:
                state = np.tile(initial_state[None, :], (group_size, 1))
                for step in range(horizon):
                    state = (
                        state @ dynamics.coef.T
                        + dynamics.intercept[None, :]
                        + innovations[mask, step]
                    )
                    paths[mask, step] = state

        return self._clip(paths)

    def _clip(self, paths: np.ndarray) -> np.ndarray:
        if self.bounds is None:
            return paths
        for index, feature in enumerate(self.features):
            if feature in self.bounds:
                lower, upper = self.bounds[feature]
                paths[..., index] = np.clip(
                    paths[..., index],
                    lower,
                    upper,
                )
        return paths

    def edge_table(self, tolerance: float = 1e-10) -> pd.DataFrame:
        """Return point coefficients and bootstrap edge-selection frequencies."""

        draws = self._require_fitted()
        rows: list[dict[str, object]] = []
        for target_index, target in enumerate(self.features):
            for parent_index, parent in enumerate(self.features):
                draw_values = np.asarray(
                    [draw.coef[target_index, parent_index] for draw in draws],
                    dtype=float,
                )
                rows.append(
                    {
                        "parent": parent,
                        "target": target,
                        "preferred_edge": parent
                        in set(self.expert_edges.get(target, ())),
                        "point_coefficient": float(
                            self.dynamics_.coef[target_index, parent_index]
                        ),
                        "selection_frequency": float(
                            np.mean(np.abs(draw_values) > tolerance)
                        ),
                        "mean_bootstrap_coefficient": float(draw_values.mean()),
                    }
                )
        return pd.DataFrame(rows)

    def diagnostics(self) -> pd.DataFrame:
        """Summarise stability and residual-pool provenance for every draw."""

        draws = self._require_fitted()
        return pd.DataFrame(
            [
                {
                    "draw": index,
                    "raw_spectral_radius": draw.raw_spectral_radius,
                    "stability_scale": draw.stability_scale,
                    "projected": draw.stability_scale < 1.0,
                    "residual_rows": len(draw.residuals),
                    "residual_pool": "chronological",
                    "maximum_absolute_residual_mean": float(
                        np.max(np.abs(draw.residuals.mean(axis=0)))
                    ),
                    "maximum_absolute_centering_shift": float(
                        np.max(np.abs(draw.residual_centering_shift))
                    ),
                    "centering_shift_l2": float(
                        np.linalg.norm(draw.residual_centering_shift)
                    ),
                    "shock_block_length": min(
                        self.block_length,
                        len(draw.residuals),
                    ),
                }
                for index, draw in enumerate(draws)
            ]
        )


GraphInformedBootstrapVAR = GIBVAR


__all__ = [
    "DEFAULT_BLOCK_LENGTH",
    "DEFAULT_BOUNDS",
    "DEFAULT_COEFFICIENT_DRAWS",
    "DEFAULT_EXPERT_EDGES",
    "DEFAULT_EXPERT_WEIGHT",
    "DEFAULT_FEATURES",
    "DEFAULT_MAX_PARENTS",
    "DEFAULT_MAX_SPECTRAL_RADIUS",
    "FittedDynamics",
    "GIBVAR",
    "GraphInformedBootstrapVAR",
    "consecutive_block_indices",
    "moving_block_pair_indices",
]
