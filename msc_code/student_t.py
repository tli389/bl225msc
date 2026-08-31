"""Conditional SSP sampling with fully adapted smoothed residual-block bridges.

The maintained SSP draws complete historical residual blocks unconditionally.
Its ordinary conditional implementation instead approximates every coefficient
system by a Gaussian path law.  This research module keeps the coefficient
systems and historical block centres, but gives every centre continuous support:

    epsilon = residual_mean
              + sqrt(1 - tau**2) * (historical_block - residual_mean)
              + kernel_noise.

The kernel noise is Gaussian, multivariate Student-t, or a covariance-matched
two-scale contaminated Gaussian.  At each block, the linear VAR recursion turns
the supplied future coordinates into linear constraints on the block
innovations.  Their conditional distribution is available analytically.  A
fully adapted particle step first samples a historical centre using its
predictive likelihood and then samples innovations from the exact kernel
conditional.  The incremental particle weight is the mixture predictive
likelihood of the supplied block.

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import gammaln, logsumexp


"""
The features/bounds/perferredgraph defaults below come from an earlier 
prototype configuration (incl. VIX). The evaluated runs never rely on them, 
every runner passes the domain's own features, graph and bounds explicitly 
(see uncon.py, cond.py, class_app/). 

"""

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


@dataclass(frozen=True)
class BlockBridgeDiagnostics:
    """Particle-support and bridge diagnostics for one conditional ensemble."""

    kernel: str
    tau: float
    degrees_of_freedom: float | None
    block_length: int
    stage_lengths: np.ndarray
    ess: np.ndarray
    ess_ratio: np.ndarray
    maximum_weight: np.ndarray
    selected_mahalanobis_per_dimension: np.ndarray
    high_scale_posterior_share: np.ndarray
    unique_selected_blocks: np.ndarray
    effective_selected_blocks: np.ndarray
    maximum_selected_block_share: np.ndarray
    resampled: np.ndarray
    unique_ancestors: int
    unique_ancestor_ratio: float
    effective_ancestors: float
    maximum_ancestor_share: float
    component_count: int
    unique_components: int
    effective_components: float
    maximum_component_share: float
    resampling_count: int
    log_predictive_evidence: float
    observed_bound_violation_fraction: float
    clipped_hidden_fraction: float
    maximum_observed_restoration_error: float

    @property
    def minimum_ess_ratio(self) -> float:
        return float(np.min(self.ess_ratio))


@dataclass(frozen=True)
class TemperedBlockBridgeDiagnostics(BlockBridgeDiagnostics):
    """Additional diagnostics emitted only by opt-in likelihood tempering."""

    tempering_enabled: bool
    tempering_target_cess_ratio: float | None
    tempering_stage: np.ndarray
    tempering_lambda_before: np.ndarray
    tempering_lambda_after: np.ndarray
    tempering_cess_ratio: np.ndarray
    tempering_ess_ratio: np.ndarray
    tempering_resampled: np.ndarray
    tempering_rejuvenated: np.ndarray


@dataclass(frozen=True)
class _BridgeComponent:
    dynamics: LinearDynamics
    block_centres: np.ndarray
    observation_projection: np.ndarray
    observation_kernel_scale: np.ndarray
    observation_precision: np.ndarray
    observation_logdet: float
    conditional_gain: np.ndarray
    conditional_root: np.ndarray


def _dynamics_components(generator: object) -> list[LinearDynamics]:
    raw = getattr(generator, "draws_", None)
    if raw is None:
        point = getattr(generator, "dynamics_", None)
        if point is None:
            raise TypeError("block-bridge conditioning requires a fitted generator")
        raw = [point]
    components = list(raw)
    if not components:
        raise ValueError("the fitted generator has no coefficient systems")
    for component in components:
        required = ("coef", "intercept", "residuals", "residual_cov")
        if any(not hasattr(component, field) for field in required):
            raise TypeError(
                "every component must provide linear dynamics and residuals"
            )
        residuals = np.asarray(component.residuals, dtype=float)
        if residuals.ndim != 2 or len(residuals) < 2:
            raise ValueError("every component needs at least two residual rows")
    return components


def _normalise_observed_path(
    supplied_path: np.ndarray,
    observed_features: list[str],
    features: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    feature_names = tuple(features)
    if not feature_names or len(set(feature_names)) != len(feature_names):
        raise ValueError("features must be non-empty and unique")
    observed = list(observed_features)
    if not observed or len(set(observed)) != len(observed):
        raise ValueError("observed_features must be non-empty and unique")
    unknown = sorted(set(observed).difference(feature_names))
    if unknown:
        raise ValueError(f"unknown observed features: {unknown}")
    if len(observed) == len(feature_names):
        raise ValueError("at least one feature must remain unobserved")

    observed_indices = np.asarray(
        [feature_names.index(name) for name in observed], dtype=int
    )
    supplied = np.asarray(supplied_path, dtype=float)
    if supplied.ndim != 2:
        raise ValueError("supplied_path must be a two-dimensional array")
    if supplied.shape[1] == len(observed):
        values = supplied
    elif supplied.shape[1] == len(feature_names):
        values = supplied[:, observed_indices]
    else:
        raise ValueError(
            "supplied_path must contain either observed columns only or all features"
        )
    if len(values) < 1 or not np.isfinite(values).all():
        raise ValueError("the supplied observed values must be non-empty and finite")
    return values, observed_indices


def _systematic_resample(
    weights: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def _normalise_log_weights(log_weights: np.ndarray) -> tuple[np.ndarray, float]:
    normaliser = float(logsumexp(log_weights))
    weights = np.exp(log_weights - normaliser)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise FloatingPointError("block-bridge particle weights are degenerate")
    weights /= weights.sum()
    return weights, normaliser


def _conditional_ess_ratio(
    weights: np.ndarray,
    log_likelihood: np.ndarray,
    increment: float,
) -> float:
    """Conditional ESS ratio for a prospective likelihood increment."""

    if increment < 0.0 or not np.isfinite(increment):
        raise ValueError("tempering increment must be finite and non-negative")
    values = np.asarray(log_likelihood, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise FloatingPointError("all block predictive likelihoods are non-finite")
    maximum = float(np.max(values[finite]))
    factors = np.exp(increment * (values - maximum))
    numerator = float(np.dot(weights, factors)) ** 2
    denominator = float(np.dot(weights, factors**2))
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise FloatingPointError("tempering conditional ESS is degenerate")
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def _next_tempering_exponent(
    current: float,
    weights: np.ndarray,
    log_likelihood: np.ndarray,
    target_ratio: float,
) -> tuple[float, float]:
    """Choose the largest next exponent satisfying the conditional ESS target."""

    remaining = 1.0 - current
    if remaining <= np.finfo(float).eps:
        return 1.0, 1.0
    full_ratio = _conditional_ess_ratio(weights, log_likelihood, remaining)
    if full_ratio >= target_ratio:
        return 1.0, full_ratio

    lower = 0.0
    upper = remaining
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        ratio = _conditional_ess_ratio(weights, log_likelihood, midpoint)
        if ratio >= target_ratio:
            lower = midpoint
        else:
            upper = midpoint
    if lower <= np.finfo(float).eps * max(1.0, remaining):
        raise FloatingPointError("adaptive tempering produced a zero-sized step")
    next_value = min(1.0, current + lower)
    achieved = _conditional_ess_ratio(
        weights,
        log_likelihood,
        next_value - current,
    )
    return next_value, achieved


def _transition_projection(
    coefficient: np.ndarray,
    length: int,
    observed_indices: np.ndarray,
) -> np.ndarray:
    """Map a vectorised innovation block to its observed state coordinates."""

    n_features = coefficient.shape[0]
    propagation = np.zeros((length * n_features, length * n_features), dtype=float)
    powers = [np.eye(n_features)]
    for _ in range(1, length):
        powers.append(coefficient @ powers[-1])
    for later in range(length):
        for earlier in range(later + 1):
            row = slice(later * n_features, (later + 1) * n_features)
            column = slice(earlier * n_features, (earlier + 1) * n_features)
            propagation[row, column] = powers[later - earlier]
    observed_flat = np.asarray(
        [
            step * n_features + feature
            for step in range(length)
            for feature in observed_indices
        ],
        dtype=int,
    )
    return propagation[observed_flat]


def _safe_inverse_and_logdet(
    covariance: np.ndarray,
) -> tuple[np.ndarray, float]:
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = max(float(np.max(eigenvalues)), 1.0)
    eigenvalues = np.maximum(eigenvalues, largest * 1e-10)
    inverse = (eigenvectors / eigenvalues[None, :]) @ eigenvectors.T
    return inverse, float(np.log(eigenvalues).sum())


def _singular_root(covariance: np.ndarray) -> np.ndarray:
    covariance = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    largest = max(float(np.max(eigenvalues)), 1.0)
    keep = eigenvalues > largest * 1e-10
    if not np.any(keep):
        return np.zeros((len(covariance), 0), dtype=float)
    return eigenvectors[:, keep] * np.sqrt(eigenvalues[keep])[None, :]


def _block_centres(residuals: np.ndarray, length: int, tau: float) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=float)
    blocks = np.stack(
        [
            residuals[start : start + length]
            for start in range(len(residuals) - length + 1)
        ]
    )
    if tau == 0.0:
        return blocks.reshape(len(blocks), -1)
    mean = residuals.mean(axis=0)
    return (
        mean[None, None, :] + np.sqrt(1.0 - tau**2) * (blocks - mean[None, None, :])
    ).reshape(len(blocks), -1)


def _build_component(
    dynamics: LinearDynamics,
    length: int,
    observed_indices: np.ndarray,
    tau: float,
    kernel: str,
    degrees_of_freedom: float,
) -> _BridgeComponent:
    coefficient = np.asarray(dynamics.coef, dtype=float)
    residual_covariance = np.asarray(dynamics.residual_cov, dtype=float)
    n_features = coefficient.shape[0]
    if residual_covariance.shape != (n_features, n_features):
        raise ValueError("component residual covariance has the wrong shape")

    centres = _block_centres(np.asarray(dynamics.residuals), length, tau)
    projection = _transition_projection(coefficient, length, observed_indices)
    noise_covariance = tau**2 * np.kron(np.eye(length), residual_covariance)
    if kernel == "student_t":
        kernel_scale = noise_covariance * (
            (degrees_of_freedom - 2.0) / degrees_of_freedom
        )
    else:
        kernel_scale = noise_covariance

    observed_scale = projection @ kernel_scale @ projection.T
    observed_precision, observed_logdet = _safe_inverse_and_logdet(observed_scale)
    cross = kernel_scale @ projection.T
    gain = cross @ observed_precision
    conditional_scale = kernel_scale - gain @ cross.T
    root = _singular_root(conditional_scale)
    return _BridgeComponent(
        dynamics=dynamics,
        block_centres=centres,
        observation_projection=projection,
        observation_kernel_scale=observed_scale,
        observation_precision=observed_precision,
        observation_logdet=observed_logdet,
        conditional_gain=gain,
        conditional_root=root,
    )


def _zero_shock_paths(
    states: np.ndarray,
    dynamics: LinearDynamics,
    length: int,
) -> np.ndarray:
    coefficient = np.asarray(dynamics.coef, dtype=float)
    intercept = np.asarray(dynamics.intercept, dtype=float)
    current = states.copy()
    paths = np.empty((len(states), length, coefficient.shape[0]), dtype=float)
    for step in range(length):
        current = current @ coefficient.T + intercept[None, :]
        paths[:, step] = current
    return paths


def _component_loglikelihoods(
    differences: np.ndarray,
    precision: np.ndarray,
    logdet: float,
    kernel: str,
    degrees_of_freedom: float,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = differences.shape[-1]
    quadratic = np.einsum(
        "...i,ij,...j->...",
        differences,
        precision,
        differences,
        optimize=True,
    )
    if kernel == "gaussian":
        constant = dimension * np.log(2.0 * np.pi) + logdet
        values = -0.5 * (constant + quadratic)
    else:
        nu = degrees_of_freedom
        constant = (
            gammaln((nu + dimension) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * (dimension * np.log(nu * np.pi) + logdet)
        )
        values = constant - 0.5 * (nu + dimension) * np.log1p(quadratic / nu)
    return values, quadratic


def _sample_categorical_rows(
    log_probabilities: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    probabilities = np.exp(
        log_probabilities - logsumexp(log_probabilities, axis=1, keepdims=True)
    )
    cumulative = np.cumsum(probabilities, axis=1)
    cumulative[:, -1] = 1.0
    uniforms = rng.random(len(probabilities))
    return np.sum(cumulative < uniforms[:, None], axis=1)


def _sample_conditional_innovations(
    component: _BridgeComponent,
    centre_indices: np.ndarray,
    differences: np.ndarray,
    selected_quadratic: np.ndarray,
    kernel: str,
    degrees_of_freedom: float,
    rng: np.random.Generator,
    variance_multipliers: np.ndarray | None = None,
) -> np.ndarray:
    centres = component.block_centres[centre_indices]
    conditional_mean = centres + differences @ component.conditional_gain.T
    rank = component.conditional_root.shape[1]
    if rank:
        normal = rng.standard_normal((len(centres), rank))
        noise = normal @ component.conditional_root.T
        if variance_multipliers is not None:
            noise *= np.sqrt(np.asarray(variance_multipliers))[:, None]
        if kernel == "student_t":
            conditional_df = degrees_of_freedom + differences.shape[1]
            chi_squared = rng.chisquare(conditional_df, size=len(centres))
            multiplier = np.sqrt(
                (degrees_of_freedom + selected_quadratic) / chi_squared
            )
            noise *= multiplier[:, None]
        innovations = conditional_mean + noise
    else:
        innovations = conditional_mean

    # Restore the exact linear constraint after floating-point factorisation.
    restoration = (
        differences - (innovations - centres) @ component.observation_projection.T
    )
    innovations += restoration @ component.conditional_gain.T
    return innovations


def _redraw_conditioned_stage(
    start_states: np.ndarray,
    assignments: np.ndarray,
    components: Sequence[LinearDynamics],
    observed_values: np.ndarray,
    observed_indices: np.ndarray,
    *,
    tau: float,
    kernel: str,
    degrees_of_freedom: float,
    contamination_probability: float,
    contamination_scale: float,
    rng: np.random.Generator,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Refresh one block from its exact locally adapted conditional proposal."""

    length = len(observed_values)
    target = observed_values.reshape(-1)
    n_particles, n_features = start_states.shape
    end_states = np.empty_like(start_states)
    stage_paths = np.empty((n_particles, length, n_features), dtype=float)
    log_predictive = np.empty(n_particles, dtype=float)
    selected_distance = np.empty(n_particles, dtype=float)
    high_scale = np.zeros(n_particles, dtype=bool)
    block_labels = np.empty((n_particles, 2), dtype=int)

    for component_index in np.unique(assignments):
        selected_particles = np.flatnonzero(assignments == component_index)
        dynamics = components[int(component_index)]
        bridge = _build_component(
            dynamics,
            length,
            observed_indices,
            tau,
            kernel,
            degrees_of_freedom,
        )
        zero_paths = _zero_shock_paths(
            start_states[selected_particles],
            dynamics,
            length,
        )
        base_observed = zero_paths[:, :, observed_indices].reshape(
            len(selected_particles),
            -1,
        )
        centre_observed = bridge.block_centres @ bridge.observation_projection.T
        differences = (
            target[None, None, :]
            - base_observed[:, None, :]
            - centre_observed[None, :, :]
        )
        loglikelihoods, quadratics = _component_loglikelihoods(
            differences,
            bridge.observation_precision,
            bridge.observation_logdet,
            "gaussian" if kernel == "contaminated_gaussian" else kernel,
            degrees_of_freedom,
        )
        variance_multipliers = None
        if kernel == "contaminated_gaussian":
            high_variance = contamination_scale**2
            low_variance = (1.0 - contamination_probability * high_variance) / (
                1.0 - contamination_probability
            )
            regime_variances = np.asarray(
                [low_variance, high_variance],
                dtype=float,
            )
            regime_log_priors = np.log(
                np.asarray(
                    [
                        1.0 - contamination_probability,
                        contamination_probability,
                    ],
                    dtype=float,
                )
            )
            dimension = differences.shape[-1]
            joint_loglikelihoods = np.stack(
                [
                    -0.5
                    * (
                        dimension * np.log(2.0 * np.pi)
                        + bridge.observation_logdet
                        + dimension * np.log(variance)
                        + quadratics / variance
                    )
                    + regime_log_priors[regime]
                    for regime, variance in enumerate(regime_variances)
                ],
                axis=-1,
            )
            predictive_loglikelihoods = joint_loglikelihoods.reshape(
                len(selected_particles),
                -1,
            )
            joint_indices = _sample_categorical_rows(
                predictive_loglikelihoods,
                rng,
            )
            centre_indices = joint_indices // len(regime_variances)
            regime_indices = joint_indices % len(regime_variances)
            variance_multipliers = regime_variances[regime_indices]
            high_scale[selected_particles] = regime_indices == 1
        else:
            centre_indices = _sample_categorical_rows(loglikelihoods, rng)
            predictive_loglikelihoods = loglikelihoods

        rows = np.arange(len(selected_particles))
        selected_differences = differences[rows, centre_indices]
        selected_quadratic = quadratics[rows, centre_indices]
        innovations = _sample_conditional_innovations(
            bridge,
            centre_indices,
            selected_differences,
            selected_quadratic,
            kernel,
            degrees_of_freedom,
            rng,
            variance_multipliers=variance_multipliers,
        ).reshape(len(selected_particles), length, n_features)

        current = start_states[selected_particles].copy()
        coefficient = np.asarray(dynamics.coef, dtype=float)
        intercept = np.asarray(dynamics.intercept, dtype=float)
        for offset in range(length):
            current = (
                current @ coefficient.T + intercept[None, :] + innovations[:, offset]
            )
            current[:, observed_indices] = observed_values[offset]
            stage_paths[selected_particles, offset] = current
        end_states[selected_particles] = current

        log_predictive[selected_particles] = logsumexp(
            predictive_loglikelihoods,
            axis=1,
        ) - np.log(bridge.block_centres.shape[0])
        selected_distance[selected_particles] = selected_quadratic / len(target)
        block_labels[selected_particles, 0] = int(component_index)
        block_labels[selected_particles, 1] = (
            0 if np.isclose(tau, 1.0) else centre_indices
        )

    return (
        end_states,
        stage_paths,
        log_predictive,
        selected_distance,
        high_scale,
        block_labels,
    )


def _resolve_schema(
    generator: object,
    features: Sequence[str] | None,
    bounds: Mapping[str, tuple[float, float]] | None,
) -> tuple[tuple[str, ...], Mapping[str, tuple[float, float]]]:
    generator_features = getattr(generator, "features", None)
    if features is None:
        feature_names = tuple(
            FEATURES if generator_features is None else generator_features
        )
    else:
        feature_names = tuple(features)
        if generator_features is not None and feature_names != tuple(
            generator_features
        ):
            raise ValueError(
                "configured feature order does not match the fitted generator"
            )
    generator_bounds = getattr(generator, "bounds", None)
    if bounds is None:
        configured_bounds = (
            SANITY_BOUNDS if generator_bounds is None else generator_bounds
        )
    else:
        configured_bounds = bounds
        if generator_bounds is not None and dict(bounds) != dict(generator_bounds):
            raise ValueError("configured bounds do not match the fitted generator")
    for feature, feature_bounds in configured_bounds.items():
        if len(feature_bounds) != 2:
            raise ValueError(f"bounds for {feature!r} must contain two values")
        lower, upper = (float(value) for value in feature_bounds)
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            raise ValueError(
                f"bounds for {feature!r} must be finite with lower < upper"
            )
    return feature_names, configured_bounds


def smoothed_block_sample(
    generator: object,
    *,
    n_paths: int,
    horizon: int,
    rng: np.random.Generator,
    jumpoff: np.ndarray,
    block_length: int = 4,
    tau: float = 0.75,
    kernel: str = "student_t",
    degrees_of_freedom: float = 20.0,
    apply_bounds: bool = False,
    features: Sequence[str] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> np.ndarray:
    """Draw the unconditional law used by the smoothed-block bridge.

    The default leaves paths unclipped so they can be used as an internally
    coherent reference sample for model-relative support calibration.
    """

    if n_paths < 1 or horizon < 1 or block_length < 1:
        raise ValueError("n_paths, horizon, and block_length must be positive")
    if not np.isfinite(tau) or not 0.0 <= tau <= 1.0:
        raise ValueError("tau must be finite and lie in [0, 1]")
    if kernel not in {"gaussian", "student_t"}:
        raise ValueError("unconditional smoothing supports Gaussian or Student-t")
    if kernel == "student_t" and (
        not np.isfinite(degrees_of_freedom) or degrees_of_freedom <= 2.0
    ):
        raise ValueError("Student-t degrees_of_freedom must exceed two")

    feature_names, configured_bounds = _resolve_schema(
        generator,
        features,
        bounds,
    )
    components = _dynamics_components(generator)
    n_features = len(feature_names)
    initial = np.asarray(jumpoff, dtype=float)
    if initial.shape != (n_features,) or not np.isfinite(initial).all():
        raise ValueError("jumpoff must be a finite vector matching features")
    assignments = rng.integers(0, len(components), size=n_paths)
    paths = np.empty((n_paths, horizon, n_features), dtype=float)

    for component_index in np.unique(assignments):
        selected = np.flatnonzero(assignments == component_index)
        dynamics = components[int(component_index)]
        expected_matrix = (n_features, n_features)
        residuals = np.asarray(dynamics.residuals, dtype=float)
        if (
            np.asarray(dynamics.coef).shape != expected_matrix
            or np.asarray(dynamics.intercept).shape != (n_features,)
            or residuals.ndim != 2
            or residuals.shape[1] != n_features
            or np.asarray(dynamics.residual_cov).shape != expected_matrix
        ):
            raise ValueError(
                "component dynamics dimension does not match configured features"
            )
        if len(residuals) < min(block_length, horizon):
            raise ValueError(
                "each component needs at least as many residual rows as the "
                "longest requested block"
            )
        states = np.repeat(initial[None, :], len(selected), axis=0)
        start = 0
        while start < horizon:
            length = min(block_length, horizon - start)
            centres = _block_centres(residuals, length, tau)
            chosen = rng.integers(0, len(centres), size=len(selected))
            if tau == 0.0:
                innovations = centres[chosen].reshape(
                    len(selected),
                    length,
                    n_features,
                )
            else:
                covariance = tau**2 * np.kron(
                    np.eye(length),
                    np.asarray(dynamics.residual_cov, dtype=float),
                )
                if kernel == "student_t":
                    covariance *= (degrees_of_freedom - 2.0) / degrees_of_freedom
                root = _singular_root(covariance)
                noise = rng.standard_normal((len(selected), root.shape[1])) @ root.T
                if kernel == "student_t":
                    noise *= np.sqrt(
                        degrees_of_freedom
                        / rng.chisquare(degrees_of_freedom, size=len(selected))
                    )[:, None]
                innovations = (centres[chosen] + noise).reshape(
                    len(selected),
                    length,
                    n_features,
                )
            for offset in range(length):
                states = (
                    states @ np.asarray(dynamics.coef, dtype=float).T
                    + np.asarray(dynamics.intercept, dtype=float)[None, :]
                    + innovations[:, offset]
                )
                paths[selected, start + offset] = states
            start += length

    if apply_bounds:
        paths = clip_to_bounds(paths, feature_names, configured_bounds)
    return paths


def block_bridge_conditional_sample(
    generator: object,
    supplied_path: np.ndarray,
    observed_features: list[str],
    *,
    n_paths: int,
    n_particles: int,
    rng: np.random.Generator,
    jumpoff: np.ndarray,
    block_length: int = 4,
    tau: float = 0.25,
    kernel: str = "student_t",
    degrees_of_freedom: float = 5.0,
    contamination_probability: float = 0.10,
    contamination_scale: float = 2.0,
    ess_resample_ratio: float = 0.5,
    likelihood_tempering: bool = False,
    tempering_cess_ratio: float = 0.80,
    max_tempering_steps: int = 64,
    rejuvenate_after_resampling: bool = True,
    apply_hidden_bounds: bool = True,
    features: Sequence[str] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, BlockBridgeDiagnostics]:
    """Sample SSP paths conditional on supplied coordinates.

    ``kernel='gaussian'`` gives a Gaussian smoothed-block bridge.
    ``kernel='student_t'`` uses a covariance-matched Student-t kernel.  The
    latter has a conditional scale that grows with the supplied block's
    Mahalanobis distance.  ``kernel='contaminated_gaussian'`` uses a
    covariance-matched ordinary/crisis scale mixture, which caps that
    conditional widening.  Optional tempering applies successive powers to
    each particle's collapsed block predictive likelihood.  Its endpoint is
    the same exact smoothed-block conditional target as the ordinary fully
    adapted step.
    """

    if n_paths < 2 or n_particles < n_paths:
        raise ValueError("require 2 <= n_paths <= n_particles")
    if block_length < 1:
        raise ValueError("block_length must be positive")
    if not np.isfinite(tau) or not 0.0 < tau <= 1.0:
        raise ValueError("tau must be finite and lie in (0, 1]")
    if kernel not in {"gaussian", "student_t", "contaminated_gaussian"}:
        raise ValueError(
            "kernel must be 'gaussian', 'student_t', or 'contaminated_gaussian'"
        )
    if kernel == "student_t" and (
        not np.isfinite(degrees_of_freedom) or degrees_of_freedom <= 2.0
    ):
        raise ValueError("Student-t degrees_of_freedom must exceed two")
    if kernel == "contaminated_gaussian":
        if (
            not np.isfinite(contamination_probability)
            or not 0.0 < contamination_probability < 1.0
        ):
            raise ValueError("contamination_probability must lie in (0, 1)")
        if not np.isfinite(contamination_scale) or contamination_scale <= 1.0:
            raise ValueError("contamination_scale must exceed one")
        if contamination_probability * contamination_scale**2 >= 1.0:
            raise ValueError("covariance matching requires probability * scale**2 < 1")
    if not 0.0 < ess_resample_ratio <= 1.0:
        raise ValueError("ess_resample_ratio must lie in (0, 1]")
    if likelihood_tempering:
        if not 0.0 < tempering_cess_ratio < 1.0:
            raise ValueError("tempering_cess_ratio must lie in (0, 1)")
        if max_tempering_steps < 1:
            raise ValueError("max_tempering_steps must be positive")

    feature_names, configured_bounds = _resolve_schema(
        generator,
        features,
        bounds,
    )
    observed_values, observed_indices = _normalise_observed_path(
        supplied_path,
        observed_features,
        feature_names,
    )
    components = _dynamics_components(generator)
    n_features = len(feature_names)
    required_residual_rows = min(block_length, len(observed_values))
    for component in components:
        expected_matrix = (n_features, n_features)
        if (
            np.asarray(component.coef).shape != expected_matrix
            or np.asarray(component.intercept).shape != (n_features,)
            or np.asarray(component.residuals).ndim != 2
            or np.asarray(component.residuals).shape[1] != n_features
            or np.asarray(component.residual_cov).shape != expected_matrix
        ):
            raise ValueError(
                "component dynamics dimension does not match configured features"
            )
        if len(np.asarray(component.residuals)) < required_residual_rows:
            raise ValueError(
                "each component needs at least as many residual rows as the "
                "longest requested bridge block"
            )
    initial = np.asarray(jumpoff, dtype=float)
    if initial.shape != (n_features,) or not np.isfinite(initial).all():
        raise ValueError("jumpoff must be a finite vector matching features")

    horizon = len(observed_values)
    assignments = rng.integers(0, len(components), size=n_particles)
    states = np.repeat(initial[None, :], n_particles, axis=0)
    paths = np.empty((n_particles, horizon, n_features), dtype=float)
    weights = np.full(n_particles, 1.0 / n_particles, dtype=float)
    ancestors = np.arange(n_particles, dtype=int)

    lengths: list[int] = []
    ess_values: list[float] = []
    maximum_weights: list[float] = []
    selected_distances: list[float] = []
    high_scale_shares: list[float] = []
    unique_blocks: list[int] = []
    effective_blocks: list[float] = []
    maximum_block_shares: list[float] = []
    resampled_stages: list[bool] = []
    tempering_stages: list[int] = []
    tempering_lambda_before: list[float] = []
    tempering_lambda_after: list[float] = []
    tempering_cess_values: list[float] = []
    tempering_ess_values: list[float] = []
    tempering_resampled: list[bool] = []
    tempering_rejuvenated: list[bool] = []
    tempering_resampling_count = 0
    log_evidence = 0.0

    start = 0
    while start < horizon:
        length = min(block_length, horizon - start)
        block_start_states = states.copy() if likelihood_tempering else None
        target = observed_values[start : start + length].reshape(-1)
        stage_log_predictive = np.empty(n_particles, dtype=float)
        stage_selected_distance = np.empty(n_particles, dtype=float)
        stage_high_scale = np.zeros(n_particles, dtype=bool)
        stage_block_labels = np.empty((n_particles, 2), dtype=int)

        for component_index in np.unique(assignments):
            selected_particles = np.flatnonzero(assignments == component_index)
            dynamics = components[int(component_index)]
            bridge = _build_component(
                dynamics,
                length,
                observed_indices,
                tau,
                kernel,
                degrees_of_freedom,
            )
            zero_paths = _zero_shock_paths(
                states[selected_particles],
                dynamics,
                length,
            )
            base_observed = zero_paths[:, :, observed_indices].reshape(
                len(selected_particles),
                -1,
            )
            centre_observed = bridge.block_centres @ bridge.observation_projection.T
            differences = (
                target[None, None, :]
                - base_observed[:, None, :]
                - centre_observed[None, :, :]
            )
            loglikelihoods, quadratics = _component_loglikelihoods(
                differences,
                bridge.observation_precision,
                bridge.observation_logdet,
                "gaussian" if kernel == "contaminated_gaussian" else kernel,
                degrees_of_freedom,
            )
            variance_multipliers = None
            if kernel == "contaminated_gaussian":
                high_variance = contamination_scale**2
                low_variance = (1.0 - contamination_probability * high_variance) / (
                    1.0 - contamination_probability
                )
                regime_variances = np.asarray(
                    [low_variance, high_variance],
                    dtype=float,
                )
                regime_log_priors = np.log(
                    np.asarray(
                        [
                            1.0 - contamination_probability,
                            contamination_probability,
                        ],
                        dtype=float,
                    )
                )
                dimension = differences.shape[-1]
                joint_loglikelihoods = np.stack(
                    [
                        -0.5
                        * (
                            dimension * np.log(2.0 * np.pi)
                            + bridge.observation_logdet
                            + dimension * np.log(variance)
                            + quadratics / variance
                        )
                        + regime_log_priors[regime]
                        for regime, variance in enumerate(regime_variances)
                    ],
                    axis=-1,
                )
                flat_loglikelihoods = joint_loglikelihoods.reshape(
                    len(selected_particles),
                    -1,
                )
                joint_indices = _sample_categorical_rows(
                    flat_loglikelihoods,
                    rng,
                )
                centre_indices = joint_indices // len(regime_variances)
                regime_indices = joint_indices % len(regime_variances)
                variance_multipliers = regime_variances[regime_indices]
                stage_high_scale[selected_particles] = regime_indices == 1
                predictive_loglikelihoods = flat_loglikelihoods
            else:
                centre_indices = _sample_categorical_rows(loglikelihoods, rng)
                predictive_loglikelihoods = loglikelihoods
            rows = np.arange(len(selected_particles))
            selected_differences = differences[rows, centre_indices]
            selected_quadratic = quadratics[rows, centre_indices]
            innovations = _sample_conditional_innovations(
                bridge,
                centre_indices,
                selected_differences,
                selected_quadratic,
                kernel,
                degrees_of_freedom,
                rng,
                variance_multipliers=variance_multipliers,
            ).reshape(len(selected_particles), length, n_features)

            current = states[selected_particles]
            coefficient = np.asarray(dynamics.coef, dtype=float)
            intercept = np.asarray(dynamics.intercept, dtype=float)
            for offset in range(length):
                current = (
                    current @ coefficient.T
                    + intercept[None, :]
                    + innovations[:, offset]
                )
                current[:, observed_indices] = observed_values[start + offset]
                paths[selected_particles, start + offset] = current
            states[selected_particles] = current

            stage_log_predictive[selected_particles] = logsumexp(
                predictive_loglikelihoods, axis=1
            ) - np.log(bridge.block_centres.shape[0])
            stage_selected_distance[selected_particles] = selected_quadratic / len(
                target
            )
            stage_block_labels[selected_particles, 0] = int(component_index)
            stage_block_labels[selected_particles, 1] = (
                0 if np.isclose(tau, 1.0) else centre_indices
            )

        if likelihood_tempering:
            stage_number = len(lengths) + 1
            temperature = 0.0
            stage_ess: list[float] = []
            stage_maximum_weights: list[float] = []
            stage_resampled = False
            stage_step = 0
            while temperature < 1.0:
                if stage_step >= max_tempering_steps:
                    raise RuntimeError(
                        "adaptive tempering exceeded max_tempering_steps"
                    )
                next_temperature, achieved_cess = _next_tempering_exponent(
                    temperature,
                    weights,
                    stage_log_predictive,
                    tempering_cess_ratio,
                )
                increment = next_temperature - temperature
                updated_weights, step_normaliser = _normalise_log_weights(
                    np.log(np.maximum(weights, np.finfo(float).tiny))
                    + increment * stage_log_predictive
                )
                log_evidence += step_normaliser
                ess = float(1.0 / np.sum(updated_weights**2))
                stage_ess.append(ess)
                stage_maximum_weights.append(float(np.max(updated_weights)))
                weights = updated_weights

                resampled = bool(
                    next_temperature < 1.0 or ess < ess_resample_ratio * n_particles
                )
                rejuvenated = False
                if resampled:
                    selected = _systematic_resample(weights, rng)
                    states = states[selected]
                    paths = paths[selected]
                    assignments = assignments[selected]
                    ancestors = ancestors[selected]
                    if block_start_states is None:
                        raise RuntimeError("missing block-start states for tempering")
                    block_start_states = block_start_states[selected]
                    stage_log_predictive = stage_log_predictive[selected]
                    stage_selected_distance = stage_selected_distance[selected]
                    stage_high_scale = stage_high_scale[selected]
                    stage_block_labels = stage_block_labels[selected]
                    weights.fill(1.0 / n_particles)
                    stage_resampled = True
                    tempering_resampling_count += 1

                    if rejuvenate_after_resampling:
                        (
                            states,
                            refreshed_paths,
                            refreshed_log_predictive,
                            stage_selected_distance,
                            stage_high_scale,
                            stage_block_labels,
                        ) = _redraw_conditioned_stage(
                            block_start_states,
                            assignments,
                            components,
                            observed_values[start : start + length],
                            observed_indices,
                            tau=tau,
                            kernel=kernel,
                            degrees_of_freedom=degrees_of_freedom,
                            contamination_probability=contamination_probability,
                            contamination_scale=contamination_scale,
                            rng=rng,
                        )
                        if not np.allclose(
                            refreshed_log_predictive,
                            stage_log_predictive,
                            rtol=1e-10,
                            atol=1e-10,
                        ):
                            raise RuntimeError(
                                "rejuvenation changed collapsed predictive weights"
                            )
                        stage_log_predictive = refreshed_log_predictive
                        paths[:, start : start + length] = refreshed_paths
                        rejuvenated = True

                tempering_stages.append(stage_number)
                tempering_lambda_before.append(float(temperature))
                tempering_lambda_after.append(float(next_temperature))
                tempering_cess_values.append(float(achieved_cess))
                tempering_ess_values.append(float(ess / n_particles))
                tempering_resampled.append(resampled)
                tempering_rejuvenated.append(rejuvenated)
                temperature = next_temperature
                stage_step += 1

            lengths.append(length)
            ess_values.append(float(np.min(stage_ess)))
            maximum_weights.append(float(np.max(stage_maximum_weights)))
            selected_distances.append(float(weights @ stage_selected_distance))
            high_scale_shares.append(
                float(weights @ stage_high_scale)
                if kernel == "contaminated_gaussian"
                else np.nan
            )
            _, block_inverse = np.unique(
                stage_block_labels,
                axis=0,
                return_inverse=True,
            )
            block_probabilities = np.bincount(
                block_inverse,
                weights=weights,
            )
            block_probabilities = block_probabilities[block_probabilities > 0.0]
            unique_blocks.append(len(block_probabilities))
            effective_blocks.append(float(1.0 / np.sum(block_probabilities**2)))
            maximum_block_shares.append(float(np.max(block_probabilities)))
            resampled_stages.append(stage_resampled)
        else:
            updated_weights, stage_normaliser = _normalise_log_weights(
                np.log(np.maximum(weights, np.finfo(float).tiny)) + stage_log_predictive
            )
            log_evidence += stage_normaliser
            ess = float(1.0 / np.sum(updated_weights**2))
            lengths.append(length)
            ess_values.append(ess)
            maximum_weights.append(float(np.max(updated_weights)))
            selected_distances.append(float(updated_weights @ stage_selected_distance))
            high_scale_shares.append(
                float(updated_weights @ stage_high_scale)
                if kernel == "contaminated_gaussian"
                else np.nan
            )
            _, block_inverse = np.unique(
                stage_block_labels,
                axis=0,
                return_inverse=True,
            )
            block_probabilities = np.bincount(
                block_inverse,
                weights=updated_weights,
            )
            block_probabilities = block_probabilities[block_probabilities > 0.0]
            unique_blocks.append(len(block_probabilities))
            effective_blocks.append(float(1.0 / np.sum(block_probabilities**2)))
            maximum_block_shares.append(float(np.max(block_probabilities)))

            weights = updated_weights
            resampled = bool(ess < ess_resample_ratio * n_particles)
            if resampled:
                selected = _systematic_resample(weights, rng)
                states = states[selected]
                paths = paths[selected]
                assignments = assignments[selected]
                ancestors = ancestors[selected]
                weights.fill(1.0 / n_particles)
            resampled_stages.append(resampled)
        start += length

    output_indices = rng.choice(
        n_particles,
        size=n_paths,
        replace=True,
        p=weights,
    )
    output = paths[output_indices].copy()
    unclipped = output.copy()
    if apply_hidden_bounds:
        output = clip_to_bounds(output, feature_names, configured_bounds)
    output[:, :, observed_indices] = observed_values[None, :, :]
    hidden_mask = np.ones(n_features, dtype=bool)
    hidden_mask[observed_indices] = False
    observed_violations = np.zeros_like(observed_values, dtype=bool)
    for column, feature_index in enumerate(observed_indices):
        feature_bounds = configured_bounds.get(feature_names[int(feature_index)])
        if feature_bounds is not None:
            lower, upper = feature_bounds
            observed_violations[:, column] = (observed_values[:, column] < lower) | (
                observed_values[:, column] > upper
            )
    clipped_fraction = float(
        np.mean(
            np.abs(output[:, :, hidden_mask] - unclipped[:, :, hidden_mask]) > 1e-12
        )
    )
    restoration_error = float(
        np.max(np.abs(output[:, :, observed_indices] - observed_values[None, :, :]))
    )

    component_probabilities = np.bincount(
        assignments,
        weights=weights,
        minlength=len(components),
    ).astype(float)
    component_probabilities /= component_probabilities.sum()
    positive_components = component_probabilities > 0.0
    ancestor_probabilities = np.bincount(
        ancestors,
        weights=weights,
        minlength=n_particles,
    ).astype(float)
    ancestor_probabilities /= ancestor_probabilities.sum()
    positive_ancestors = ancestor_probabilities > 0.0
    diagnostic_fields = dict(
        kernel=kernel,
        tau=float(tau),
        degrees_of_freedom=(
            float(degrees_of_freedom) if kernel == "student_t" else None
        ),
        block_length=int(block_length),
        stage_lengths=np.asarray(lengths, dtype=int),
        ess=np.asarray(ess_values, dtype=float),
        ess_ratio=np.asarray(ess_values, dtype=float) / n_particles,
        maximum_weight=np.asarray(maximum_weights, dtype=float),
        selected_mahalanobis_per_dimension=np.asarray(
            selected_distances,
            dtype=float,
        ),
        high_scale_posterior_share=np.asarray(
            high_scale_shares,
            dtype=float,
        ),
        unique_selected_blocks=np.asarray(unique_blocks, dtype=int),
        effective_selected_blocks=np.asarray(effective_blocks, dtype=float),
        maximum_selected_block_share=np.asarray(
            maximum_block_shares,
            dtype=float,
        ),
        resampled=np.asarray(resampled_stages, dtype=bool),
        unique_ancestors=int(np.sum(positive_ancestors)),
        unique_ancestor_ratio=float(np.mean(positive_ancestors)),
        effective_ancestors=float(
            1.0 / np.sum(ancestor_probabilities[positive_ancestors] ** 2)
        ),
        maximum_ancestor_share=float(np.max(ancestor_probabilities)),
        component_count=len(components),
        unique_components=int(np.sum(positive_components)),
        effective_components=float(
            1.0 / np.sum(component_probabilities[positive_components] ** 2)
        ),
        maximum_component_share=float(np.max(component_probabilities)),
        resampling_count=(
            int(tempering_resampling_count)
            if likelihood_tempering
            else int(np.sum(resampled_stages))
        ),
        log_predictive_evidence=float(log_evidence),
        observed_bound_violation_fraction=float(np.mean(observed_violations)),
        clipped_hidden_fraction=clipped_fraction,
        maximum_observed_restoration_error=restoration_error,
    )
    if likelihood_tempering:
        diagnostics = TemperedBlockBridgeDiagnostics(
            **diagnostic_fields,
            tempering_enabled=True,
            tempering_target_cess_ratio=float(tempering_cess_ratio),
            tempering_stage=np.asarray(tempering_stages, dtype=int),
            tempering_lambda_before=np.asarray(
                tempering_lambda_before,
                dtype=float,
            ),
            tempering_lambda_after=np.asarray(
                tempering_lambda_after,
                dtype=float,
            ),
            tempering_cess_ratio=np.asarray(tempering_cess_values, dtype=float),
            tempering_ess_ratio=np.asarray(tempering_ess_values, dtype=float),
            tempering_resampled=np.asarray(tempering_resampled, dtype=bool),
            tempering_rejuvenated=np.asarray(tempering_rejuvenated, dtype=bool),
        )
    else:
        diagnostics = BlockBridgeDiagnostics(**diagnostic_fields)
    return output, diagnostics


__all__ = [
    "BlockBridgeDiagnostics",
    "TemperedBlockBridgeDiagnostics",
    "block_bridge_conditional_sample",
    "smoothed_block_sample",
]
