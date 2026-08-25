"""Gaussian path conditioning for fitted linear scenario generators.

The promoted parameter-uncertainty generator is an empirical mixture of
fitted linear dynamics.  Under the Gaussian conditional approximation, each
coefficient draw defines one joint path distribution.  Supplying (for
example) the future unemployment path updates the mixture weights and the
remaining variables are sampled from the corresponding conditional laws.

This module owns that approximation.  Native unconditional generation is
unchanged and may continue to use empirical residual blocks.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

# ---- inlined support definitions (verbatim from core.py, data.py and
# ---- generators.py of the gibvar package) so this file stands alone.
# ---- Everything below the end marker is byte-identical to the original.

SANITY_BOUNDS = {
    "gdp_growth": (-15.0, 12.0),
    "fed_funds": (0.0, 20.0),
    "mortgage_30y": (0.0, 20.0),
    "bbb_spread": (0.0, 12.0),
    "unemployment": (0.0, 25.0),
    "hpi_qoq_growth": (-15.0, 10.0),
    "vix": (5.0, 120.0),
}

FEATURES = [
    "gdp_growth",
    "fed_funds",
    "mortgage_30y",
    "bbb_spread",
    "unemployment",
    "hpi_qoq_growth",
    "vix",
]


def clip_to_bounds(
    paths: np.ndarray,
    features: Sequence[str],
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Apply configured feature bounds to a path array in place."""
    configured_bounds = SANITY_BOUNDS if bounds is None else bounds
    for index, feature in enumerate(features):
        if feature in configured_bounds:
            lo, hi = configured_bounds[feature]
            paths[..., index] = np.clip(paths[..., index], lo, hi)
    return paths


@dataclass
class LinearDynamics:
    """Fitted lag-1 linear system y_t = c + A y_{t-1} + e_t."""

    coef: np.ndarray  # (k, k), row = target equation
    intercept: np.ndarray  # (k,)
    residuals: np.ndarray  # (n, k)
    residual_cov: np.ndarray  # (k, k)
    raw_spectral_radius: float = np.nan
    stability_scale: float = 1.0
# ---- end of inlined support; the original module continues verbatim ----

SUPPORT_DISTANCE_WARNING = 4.0


@dataclass(frozen=True)
class ConditionalMixtureDiagnostics:
    """Diagnostics for mixture updating by the supplied path."""

    loglikelihoods: np.ndarray
    posterior_weights: np.ndarray
    component_count: int
    effective_components: float
    effective_component_ratio: float
    maximum_weight: float
    log_mixture_evidence: float
    minimum_mahalanobis_per_dimension: float
    weight_collapse_warning: bool
    distance_support_warning: bool
    combined_support_warning: bool
    support_warning: bool


@dataclass(frozen=True)
class _ConditionalComponent:
    hidden: np.ndarray
    observed: np.ndarray
    observed_values: np.ndarray
    conditional_mean: np.ndarray
    conditional_factor: np.ndarray
    observed_loglik: float
    mahalanobis_per_dimension: float


def validate_observed_features(
    observed_features: list[str],
    features: Sequence[str] | None = None,
) -> list[str]:
    """Return a validated copy of the supplied feature names."""
    observed = list(observed_features)
    feature_names = tuple(FEATURES if features is None else features)
    if not observed:
        raise ValueError("observe at least one feature")
    unknown = [name for name in observed if name not in feature_names]
    if unknown:
        raise ValueError(f"unknown observed features: {unknown}")
    duplicates = sorted({name for name in observed if observed.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate observed features: {duplicates}")
    if len(observed) == len(feature_names):
        raise ValueError("cannot observe every feature; none would remain to score")
    return observed


def _validate_supplied_path(
    supplied_path: np.ndarray,
    observed_features: list[str],
    features: Sequence[str] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    feature_names = tuple(FEATURES if features is None else features)
    configured_bounds = SANITY_BOUNDS if bounds is None else bounds
    supplied = np.asarray(supplied_path, dtype=float)
    if supplied.ndim != 2 or supplied.shape[1] != len(feature_names):
        raise ValueError("supplied_path must have shape (horizon, len(features))")
    if supplied.shape[0] < 1:
        raise ValueError("supplied_path must contain at least one horizon")
    if not np.isfinite(supplied).all():
        raise ValueError("supplied_path must contain only finite values")
    for feature in observed_features:
        index = feature_names.index(feature)
        lower, upper = configured_bounds[feature]
        values = supplied[:, index]
        if np.any((values < lower) | (values > upper)):
            raise ValueError(
                f"observed path for {feature!r} leaves configured bounds "
                f"[{lower}, {upper}]"
            )
    return supplied


def _positive_eigendecomposition(
    covariance: np.ndarray,
    relative_floor: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetrise and floor a covariance before solving or sampling."""
    covariance = 0.5 * (covariance + covariance.T)
    values, vectors = np.linalg.eigh(covariance)
    largest = max(float(np.max(values)), 1.0)
    values = np.maximum(values, relative_floor * largest)
    return values, vectors


def joint_gaussian_moments(
    dynamics: LinearDynamics,
    jumpoff: np.ndarray,
    horizon: int,
    features: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Mean and covariance of a stacked VAR(1) path conditional on x_0."""
    coefficient = np.asarray(dynamics.coef, dtype=float)
    intercept = np.asarray(dynamics.intercept, dtype=float)
    innovation_covariance = np.asarray(dynamics.residual_cov, dtype=float)
    feature_names = tuple(FEATURES if features is None else features)
    n_features = len(feature_names)
    expected_shape = (n_features, n_features)
    if (
        coefficient.shape != expected_shape
        or innovation_covariance.shape != expected_shape
    ):
        raise ValueError("linear dynamics feature dimension does not match FEATURES")
    if intercept.shape != (n_features,):
        raise ValueError("linear dynamics intercept dimension does not match FEATURES")
    previous = np.asarray(jumpoff, dtype=float)
    if previous.shape != (n_features,) or not np.isfinite(previous).all():
        raise ValueError("jumpoff must be a finite vector matching FEATURES")

    means = np.empty((horizon, n_features), dtype=float)
    for step in range(horizon):
        previous = coefficient @ previous + intercept
        means[step] = previous

    marginal_covariances: list[np.ndarray] = []
    marginal = np.zeros(expected_shape, dtype=float)
    for _ in range(horizon):
        marginal = coefficient @ marginal @ coefficient.T + innovation_covariance
        marginal_covariances.append(marginal.copy())

    joint = np.zeros((horizon * n_features, horizon * n_features), dtype=float)
    for earlier in range(horizon):
        earlier_slice = slice(
            earlier * n_features,
            (earlier + 1) * n_features,
        )
        joint[earlier_slice, earlier_slice] = marginal_covariances[earlier]
        cross = marginal_covariances[earlier]
        for later in range(earlier + 1, horizon):
            cross = coefficient @ cross
            later_slice = slice(later * n_features, (later + 1) * n_features)
            joint[later_slice, earlier_slice] = cross
            joint[earlier_slice, later_slice] = cross.T
    return means.reshape(-1), joint


def _conditional_component(
    dynamics: LinearDynamics,
    supplied_path: np.ndarray,
    observed_features: list[str],
    jumpoff: np.ndarray,
    features: Sequence[str] | None = None,
) -> _ConditionalComponent:
    feature_names = tuple(FEATURES if features is None else features)
    horizon, n_features = supplied_path.shape
    feature_indices = [feature_names.index(feature) for feature in observed_features]
    mean, covariance = joint_gaussian_moments(
        dynamics,
        jumpoff,
        horizon,
        feature_names,
    )
    observed = np.asarray(
        [
            step * n_features + feature
            for step in range(horizon)
            for feature in feature_indices
        ],
        dtype=int,
    )
    hidden_mask = np.ones(horizon * n_features, dtype=bool)
    hidden_mask[observed] = False
    hidden = np.flatnonzero(hidden_mask)
    observed_values = supplied_path[:, feature_indices].reshape(-1)

    observed_covariance = covariance[np.ix_(observed, observed)]
    hidden_observed = covariance[np.ix_(hidden, observed)]
    hidden_covariance = covariance[np.ix_(hidden, hidden)]
    observed_values_eig, observed_vectors = _positive_eigendecomposition(
        observed_covariance
    )
    centered = observed_values - mean[observed]
    rotated = observed_vectors.T @ centered
    quadratic = float(np.sum(rotated**2 / observed_values_eig))
    log_determinant = float(np.log(observed_values_eig).sum())
    observed_loglik = -0.5 * (
        len(observed) * np.log(2.0 * np.pi) + log_determinant + quadratic
    )

    projected = hidden_observed @ observed_vectors
    gain = (projected / observed_values_eig[None, :]) @ observed_vectors.T
    conditional_mean = mean[hidden] + gain @ centered
    conditional_covariance = hidden_covariance - gain @ hidden_observed.T
    conditional_values, conditional_vectors = _positive_eigendecomposition(
        conditional_covariance
    )
    conditional_factor = conditional_vectors * np.sqrt(conditional_values)
    return _ConditionalComponent(
        hidden=hidden,
        observed=observed,
        observed_values=observed_values,
        conditional_mean=conditional_mean,
        conditional_factor=conditional_factor,
        observed_loglik=float(observed_loglik),
        mahalanobis_per_dimension=float(quadratic / len(observed)),
    )


def _dynamics_components(generator: object) -> list[LinearDynamics]:
    if getattr(generator, "supports_gaussian_conditioning", True) is False:
        name = getattr(generator, "name", type(generator).__name__)
        raise TypeError(
            "constant-covariance Gaussian conditioning is invalid for "
            f"{name}; use a native stochastic-volatility conditioner"
        )
    point = getattr(generator, "dynamics_", None)
    if point is None:
        name = getattr(generator, "name", type(generator).__name__)
        raise TypeError(
            f"observe semantics are unsupported for {name!r}: "
            "fitted linear dynamics are required"
        )
    draws = getattr(generator, "draws_", None)
    components = list(draws) if draws is not None and len(draws) else [point]
    required = ("coef", "intercept", "residual_cov")
    for component in components:
        if any(not hasattr(component, field) for field in required):
            raise TypeError("all conditional components must be LinearDynamics-like")
    return components


def gaussian_mixture_sample(
    generator: object,
    n_paths: int,
    horizon: int,
    rng: np.random.Generator,
    jumpoff: np.ndarray,
    *,
    features: Sequence[str] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    apply_bounds: bool = True,
) -> np.ndarray:
    """Sample the unconditional Gaussian law of fitted dynamics components.

    This is the matched reference for measuring the information gained from a
    supplied path.  It uses the same coefficient components and Gaussian path
    approximation as :func:`conditional_mixture_sample`, but retains the
    equal-weight component prior and supplies no future observations.  Native
    unconditional generation is deliberately unchanged and may continue to
    use empirical residual blocks.
    """
    if n_paths < 1 or horizon < 1:
        raise ValueError("n_paths and horizon must be positive")
    feature_names = tuple(FEATURES if features is None else features)
    configured_bounds = SANITY_BOUNDS if bounds is None else bounds
    components = _dynamics_components(generator)
    assignments = rng.integers(0, len(components), size=n_paths)
    n_features = len(feature_names)
    flat_paths = np.empty((n_paths, horizon * n_features), dtype=float)
    for component_index in np.unique(assignments):
        selected = np.flatnonzero(assignments == component_index)
        mean, covariance = joint_gaussian_moments(
            components[int(component_index)],
            jumpoff,
            horizon,
            feature_names,
        )
        values, vectors = _positive_eigendecomposition(covariance)
        factor = vectors * np.sqrt(values)
        standard_normal = rng.standard_normal((len(selected), len(mean)))
        flat_paths[selected] = mean[None, :] + standard_normal @ factor.T
    paths = flat_paths.reshape(n_paths, horizon, n_features)
    if apply_bounds:
        paths = clip_to_bounds(paths, feature_names, configured_bounds)
    return paths


def conditional_mixture_sample(
    generator: object,
    supplied_path: np.ndarray,
    observed_features: list[str],
    n_paths: int,
    rng: np.random.Generator,
    jumpoff: np.ndarray,
    *,
    features: Sequence[str] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    apply_hidden_bounds: bool = True,
) -> tuple[np.ndarray, ConditionalMixtureDiagnostics]:
    """Sample a Gaussian path conditional over fitted dynamics components.

    Deterministic generators contribute one component.  Generators exposing
    ``draws_`` contribute an equally weighted empirical component mixture;
    the supplied path updates those weights by marginal likelihood.
    """
    feature_names = tuple(FEATURES if features is None else features)
    configured_bounds = SANITY_BOUNDS if bounds is None else bounds
    observed = validate_observed_features(observed_features, feature_names)
    supplied = _validate_supplied_path(
        supplied_path,
        observed,
        feature_names,
        configured_bounds,
    )
    if n_paths < 1:
        raise ValueError("n_paths must be positive")
    components = [
        _conditional_component(
            dynamics,
            supplied,
            observed,
            jumpoff,
            feature_names,
        )
        for dynamics in _dynamics_components(generator)
    ]
    loglikelihoods = np.asarray(
        [component.observed_loglik for component in components],
        dtype=float,
    )
    maximum_loglikelihood = float(np.max(loglikelihoods))
    unnormalised = np.exp(loglikelihoods - maximum_loglikelihood)
    posterior_weights = unnormalised / float(unnormalised.sum())
    assignments = rng.choice(
        len(components),
        size=n_paths,
        p=posterior_weights,
    )

    horizon, n_features = supplied.shape
    flat_paths = np.empty((n_paths, horizon * n_features), dtype=float)
    for component_index in np.unique(assignments):
        selected = np.flatnonzero(assignments == component_index)
        component = components[int(component_index)]
        standard_normal = rng.standard_normal((len(selected), len(component.hidden)))
        flat_paths[np.ix_(selected, component.hidden)] = (
            component.conditional_mean[None, :]
            + standard_normal @ component.conditional_factor.T
        )
        flat_paths[np.ix_(selected, component.observed)] = component.observed_values[
            None, :
        ]
    paths = flat_paths.reshape(n_paths, horizon, n_features)
    if apply_hidden_bounds:
        paths = clip_to_bounds(paths, feature_names, configured_bounds)
    for feature in observed:
        index = feature_names.index(feature)
        paths[:, :, index] = supplied[None, :, index]

    minimum_distance = float(
        min(component.mahalanobis_per_dimension for component in components)
    )
    component_count = len(components)
    effective_components = float(1.0 / np.sum(posterior_weights**2))
    maximum_weight = float(np.max(posterior_weights))
    weight_collapse = bool(
        component_count > 1
        and (
            effective_components < max(1.5, 0.10 * component_count)
            or maximum_weight > 0.90
        )
    )
    distance_warning = minimum_distance > SUPPORT_DISTANCE_WARNING
    diagnostics = ConditionalMixtureDiagnostics(
        loglikelihoods=loglikelihoods,
        posterior_weights=posterior_weights,
        component_count=component_count,
        effective_components=effective_components,
        effective_component_ratio=effective_components / component_count,
        maximum_weight=maximum_weight,
        log_mixture_evidence=float(
            maximum_loglikelihood + np.log(float(unnormalised.mean()))
        ),
        minimum_mahalanobis_per_dimension=minimum_distance,
        weight_collapse_warning=weight_collapse,
        distance_support_warning=distance_warning,
        combined_support_warning=weight_collapse or distance_warning,
        # Backwards-compatible meaning: distance from every fitted component.
        support_warning=distance_warning,
    )
    return paths, diagnostics


__all__ = [
    "ConditionalMixtureDiagnostics",
    "SUPPORT_DISTANCE_WARNING",
    "conditional_mixture_sample",
    "gaussian_mixture_sample",
    "joint_gaussian_moments",
    "validate_observed_features",
]
