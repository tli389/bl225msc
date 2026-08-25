"""Matched posterior-predictive VAR generators for the thesis evaluation.

The two public generators in this module share the same VAR(1) likelihood,
inverse-Wishart innovation prior, semi-conjugate Gibbs sampler, stability rule,
and Gaussian predictive shock law.  They differ only in their coefficient
prior:

``MinnesotaPosteriorVARGenerator``
    Every cross-lag receives the same Minnesota shrinkage.

``SoftStructuralPosteriorVARGenerator``
    Expert-supported cross-lags receive a wider Gaussian prior, but remain
    free to shrink to zero.  Unsupported cross-lags remain available.

The coefficient prior is independent of the innovation covariance.  This is
therefore an independent Normal--inverse-Wishart, semi-conjugate BVAR rather
than a natural-conjugate matrix-normal BVAR.  Its Gibbs conditionals are exact
for the declared model.  Each predictive path retains one paired coefficient
and covariance draw for its full horizon.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import invwishart

# ---- inlined support definitions (verbatim from core.py and data.py of
# ---- the gibvar package) so this file stands alone.  Everything below the
# ---- end marker is byte-identical to the original module.

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
# ---- end of inlined support; the original module continues verbatim ----

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


EXPERT_EDGES: dict[str, list[str]] = {
    "gdp_growth": ["fed_funds"],
    "fed_funds": ["unemployment"],
    "mortgage_30y": ["fed_funds"],
    "bbb_spread": ["vix"],
    "unemployment": ["gdp_growth", "bbb_spread"],
    "hpi_qoq_growth": ["unemployment", "mortgage_30y"],
    "vix": ["bbb_spread"],
}
# ---- end of inlined support (part 2); original module continues ----

FULL_BVAR_NAME = "full_bvar_gaussian"
FULL_SSP_NAME = "full_ssp_gaussian"


@dataclass(frozen=True)
class _StandardizedSystem:
    design: np.ndarray
    targets: np.ndarray
    x: np.ndarray
    y: np.ndarray
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray


@dataclass(frozen=True)
class _PosteriorState:
    coefficient_vector: np.ndarray
    standardized_covariance: np.ndarray
    dynamics: LinearDynamics


def _symmetric_positive_definite(
    matrix: np.ndarray,
    floor: float = 1e-10,
) -> np.ndarray:
    """Return a symmetric positive-definite approximation."""
    symmetric = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix).T)
    values, vectors = np.linalg.eigh(symmetric)
    values = np.maximum(values, floor)
    return (vectors * values[None, :]) @ vectors.T


def _standardized_system(
    history: pd.DataFrame,
    features: Sequence[str] | None = None,
) -> _StandardizedSystem:
    feature_names = tuple(FEATURES if features is None else features)
    missing = [feature for feature in feature_names if feature not in history.columns]
    if missing:
        raise ValueError(f"history is missing required features: {missing}")
    values = history[list(feature_names)].to_numpy(dtype=float)
    if len(values) < 24:
        raise ValueError("at least 24 quarters are required")
    if not np.isfinite(values).all():
        raise ValueError("history contains non-finite values")

    x = values[:-1]
    y = values[1:]
    x_mean = x.mean(axis=0)
    x_scale = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
    y_mean = y.mean(axis=0)
    y_scale = np.where(y.std(axis=0) > 1e-8, y.std(axis=0), 1.0)
    design = np.column_stack([np.ones(len(x)), (x - x_mean) / x_scale])
    targets = (y - y_mean) / y_scale
    return _StandardizedSystem(
        design=design,
        targets=targets,
        x=x,
        y=y,
        x_mean=x_mean,
        x_scale=x_scale,
        y_mean=y_mean,
        y_scale=y_scale,
    )


def _draw_gaussian_from_precision(
    precision: np.ndarray,
    right_hand: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Gaussian mean and one draw from a precision matrix."""
    precision = 0.5 * (precision + precision.T)
    scale = max(float(np.max(np.diag(precision))), 1.0)
    jitter = 0.0
    for _ in range(6):
        try:
            root = np.linalg.cholesky(precision + np.eye(precision.shape[0]) * jitter)
            break
        except np.linalg.LinAlgError:
            jitter = 1e-12 * scale if jitter == 0.0 else jitter * 10.0
    else:
        raise RuntimeError("coefficient posterior precision is not positive definite")

    mean = np.linalg.solve(root.T, np.linalg.solve(root, right_hand))
    draw = mean + np.linalg.solve(root.T, rng.standard_normal(len(mean)))
    return mean, draw


def _lag1_diagnostics(values: np.ndarray) -> tuple[float, float]:
    """Return lag-one autocorrelation and its AR(1) ESS approximation."""
    values = np.asarray(values, dtype=float)
    n_values = len(values)
    if n_values < 3 or float(np.var(values)) <= 1e-14:
        return 0.0, float(n_values)
    correlation = float(np.corrcoef(values[:-1], values[1:])[0, 1])
    if not np.isfinite(correlation):
        return 0.0, float(n_values)
    correlation = float(np.clip(correlation, -0.99, 0.99))
    effective = n_values * (1.0 - correlation) / (1.0 + correlation)
    return correlation, float(np.clip(effective, 1.0, n_values))


class FullPosteriorVARGenerator:
    """Full posterior-predictive Gaussian VAR(1) with fixed proper priors.

    The base class implements the common sampler.  Passing ``expert_weight``
    activates the soft structural prior; ``None`` gives ordinary Minnesota
    cross-lag shrinkage.  The two named subclasses are preferred in evaluation
    code because their model identities are explicit.
    """

    name = "full_posterior_var_gaussian"
    shock_mode = "posterior_gaussian"

    def __init__(
        self,
        n_dynamics: int = 100,
        delta: float = 0.9,
        own_scale: float = 0.2,
        cross_scale: float = 0.1,
        intercept_scale: float = 10.0,
        covariance_prior_df: float | None = None,
        burn_in: int = 150,
        thin: int = 3,
        max_spectral_radius: float = 0.995,
        max_sampling_iterations: int = 20000,
        seed: int = 211,
        *,
        expert_weight: float | None = None,
        features: Sequence[str] | None = None,
        expert_edges: Mapping[str, Sequence[str]] | None = None,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        own_lag_means: Mapping[str, float] | None = None,
    ) -> None:
        if int(n_dynamics) < 1:
            raise ValueError("n_dynamics must be positive")
        if own_scale <= 0.0 or cross_scale <= 0.0 or intercept_scale <= 0.0:
            raise ValueError("coefficient prior scales must be positive")
        if int(burn_in) < 0 or int(thin) < 1:
            raise ValueError("burn_in must be non-negative and thin must be positive")
        if not 0.0 < max_spectral_radius < 1.0:
            raise ValueError("max_spectral_radius must lie in (0, 1)")
        if int(max_sampling_iterations) <= int(burn_in):
            raise ValueError("max_sampling_iterations must exceed burn_in")
        if expert_weight is not None and not 0.0 < expert_weight <= 1.0:
            raise ValueError("expert_weight must lie in (0, 1]")

        self.n_dynamics = int(n_dynamics)
        self.delta = float(delta)
        self.own_scale = float(own_scale)
        self.cross_scale = float(cross_scale)
        self.intercept_scale = float(intercept_scale)
        self.covariance_prior_df = covariance_prior_df
        self.burn_in = int(burn_in)
        self.thin = int(thin)
        self.max_spectral_radius = float(max_spectral_radius)
        self.max_sampling_iterations = int(max_sampling_iterations)
        self.seed = int(seed)
        self.expert_weight = None if expert_weight is None else float(expert_weight)
        self.features = tuple(FEATURES if features is None else features)
        if not self.features or len(set(self.features)) != len(self.features):
            raise ValueError("features must be non-empty and unique")
        self.expert_edges = EXPERT_EDGES if expert_edges is None else dict(expert_edges)
        self.bounds = SANITY_BOUNDS if bounds is None else dict(bounds)
        self.own_lag_means = {} if own_lag_means is None else dict(own_lag_means)
        unknown_prior_features = set(self.own_lag_means).difference(self.features)
        if unknown_prior_features:
            raise ValueError(
                "own_lag_means contains unknown features: "
                f"{sorted(unknown_prior_features)}"
            )
        if not all(np.isfinite(value) for value in self.own_lag_means.values()):
            raise ValueError("own_lag_means must contain only finite values")

        self.dynamics_: LinearDynamics | None = None
        self.draws_: list[LinearDynamics] | None = None
        self.coefficient_draws_: np.ndarray | None = None
        self.standardized_covariance_draws_: np.ndarray | None = None
        self.posterior_mean_: np.ndarray | None = None
        self.posterior_covariance_mean_: np.ndarray | None = None
        self.prior_mean_vector_: np.ndarray | None = None
        self.prior_sd_vector_: np.ndarray | None = None
        self.gibbs_iterations_: int = 0
        self.unstable_rejections_: int = 0
        self.representative_draw_index_: int | None = None
        self.last_draw_assignments_: np.ndarray | None = None
        self.additional_posterior_unstable_rejections_: int = 0
        self._fitted_system_: _StandardizedSystem | None = None
        self._fitted_prior_mean_: np.ndarray | None = None
        self._fitted_prior_precision_: np.ndarray | None = None
        self._fitted_prior_df_: float | None = None
        self._fitted_prior_scale_: np.ndarray | None = None
        self._continuation_covariance_: np.ndarray | None = None

    # --------------------------------------------------------------- priors
    def _prior_specification(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_features = len(self.features)
        prior_mean = np.zeros((n_features + 1, n_features), dtype=float)
        prior_sd = np.full(
            (n_features + 1, n_features),
            self.cross_scale,
            dtype=float,
        )
        labels = np.full(
            (n_features + 1, n_features),
            "cross",
            dtype=object,
        )
        prior_sd[0] = self.intercept_scale
        labels[0] = "intercept"

        for target, target_name in enumerate(self.features):
            prior_mean[target + 1, target] = self.own_lag_means.get(
                target_name,
                self.delta,
            )
            prior_sd[target + 1, target] = self.own_scale
            labels[target + 1, target] = "own"
            if self.expert_weight is None:
                continue
            expert_parents = set(self.expert_edges.get(target_name, ()))
            for parent, parent_name in enumerate(self.features):
                if parent == target or parent_name not in expert_parents:
                    continue
                prior_sd[parent + 1, target] = self.cross_scale / np.sqrt(
                    self.expert_weight
                )
                labels[parent + 1, target] = "expert"

        if self.expert_weight is not None:
            labels[labels == "cross"] = "other"
        return prior_mean, prior_sd, labels

    def prior_diagnostics(self) -> pd.DataFrame:
        """Return the declared coefficient prior for every equation and lag."""
        prior_mean, prior_sd, labels = self._prior_specification()
        rows: list[dict[str, object]] = []
        for target, target_name in enumerate(self.features):
            for row in range(len(self.features) + 1):
                parent = "intercept" if row == 0 else self.features[row - 1]
                rows.append(
                    {
                        "generator": self.name,
                        "target": target_name,
                        "parent": parent,
                        "group": str(labels[row, target]),
                        "prior_mean": float(prior_mean[row, target]),
                        "prior_sd": float(prior_sd[row, target]),
                        "prior_precision": float(1.0 / prior_sd[row, target] ** 2),
                    }
                )
        return pd.DataFrame(rows)

    # ---------------------------------------------------------- conditionals
    @staticmethod
    def _draw_coefficients(
        system: _StandardizedSystem,
        covariance: np.ndarray,
        prior_mean: np.ndarray,
        prior_precision: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        inverse_covariance = np.linalg.inv(_symmetric_positive_definite(covariance))
        gram = system.design.T @ system.design
        precision = np.kron(inverse_covariance, gram) + np.diag(prior_precision)
        right_hand = (system.design.T @ system.targets @ inverse_covariance).reshape(
            -1, order="F"
        ) + prior_precision * prior_mean
        _, draw = _draw_gaussian_from_precision(precision, right_hand, rng)
        return draw

    @staticmethod
    def _draw_covariance(
        system: _StandardizedSystem,
        coefficient_vector: np.ndarray,
        prior_df: float,
        prior_scale: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        coefficients = coefficient_vector.reshape(
            system.design.shape[1],
            system.targets.shape[1],
            order="F",
        )
        residuals = system.targets - system.design @ coefficients
        scale = _symmetric_positive_definite(prior_scale + residuals.T @ residuals)
        covariance = invwishart.rvs(
            df=prior_df + len(system.targets),
            scale=scale,
            random_state=rng,
        )
        return np.atleast_2d(covariance)

    @staticmethod
    def _to_raw_dynamics(
        coefficient_vector: np.ndarray,
        standardized_covariance: np.ndarray,
        system: _StandardizedSystem,
    ) -> LinearDynamics:
        n_features = system.targets.shape[1]
        standardized = coefficient_vector.reshape(
            n_features + 1,
            n_features,
            order="F",
        )
        coefficient = (
            standardized[1:].T * system.y_scale[:, None] / system.x_scale[None, :]
        )
        intercept = (
            system.y_mean
            + system.y_scale * standardized[0]
            - coefficient @ system.x_mean
        )
        radius = float(np.max(np.abs(np.linalg.eigvals(coefficient))))
        residuals = system.y - (system.x @ coefficient.T + intercept[None, :])
        rescale = np.diag(system.y_scale)
        raw_covariance = _symmetric_positive_definite(
            rescale @ standardized_covariance @ rescale
        )
        return LinearDynamics(
            coef=coefficient,
            intercept=intercept,
            residuals=residuals,
            residual_cov=raw_covariance,
            raw_spectral_radius=radius,
            stability_scale=1.0,
        )

    # ------------------------------------------------------------------ fit
    def fit(self, history: pd.DataFrame) -> FullPosteriorVARGenerator:
        system = _standardized_system(history, self.features)
        prior_mean_matrix, prior_sd_matrix, _ = self._prior_specification()
        prior_mean = prior_mean_matrix.reshape(-1, order="F")
        prior_sd = prior_sd_matrix.reshape(-1, order="F")
        prior_precision = 1.0 / prior_sd**2
        self.prior_mean_vector_ = prior_mean.copy()
        self.prior_sd_vector_ = prior_sd.copy()

        n_features = len(self.features)
        prior_df = (
            float(self.covariance_prior_df)
            if self.covariance_prior_df is not None
            else float(n_features + 2)
        )
        if prior_df <= n_features + 1:
            raise ValueError("covariance_prior_df must exceed n_features + 1")
        prior_scale = np.eye(n_features) * (prior_df - n_features - 1.0)

        # Retain the fixed training-posterior specification so conditional
        # scenario expansion can enlarge a thin empirical state reservoir
        # without refitting or changing the target distribution.
        self._fitted_system_ = system
        self._fitted_prior_mean_ = prior_mean.copy()
        self._fitted_prior_precision_ = prior_precision.copy()
        self._fitted_prior_df_ = prior_df
        self._fitted_prior_scale_ = prior_scale.copy()

        rng = np.random.default_rng(self.seed + len(history))
        covariance = np.eye(n_features)
        retained: list[_PosteriorState] = []
        unstable_rejections = 0
        iteration = 0

        while len(retained) < self.n_dynamics:
            iteration += 1
            if iteration > self.max_sampling_iterations:
                raise RuntimeError(
                    "insufficient stable posterior states; "
                    f"retained {len(retained)} of {self.n_dynamics} after "
                    f"{self.max_sampling_iterations} Gibbs iterations"
                )
            coefficient_vector = self._draw_coefficients(
                system,
                covariance,
                prior_mean,
                prior_precision,
                rng,
            )
            covariance = self._draw_covariance(
                system,
                coefficient_vector,
                prior_df,
                prior_scale,
                rng,
            )

            if iteration <= self.burn_in:
                continue
            if (iteration - self.burn_in - 1) % self.thin != 0:
                continue
            dynamics = self._to_raw_dynamics(
                coefficient_vector,
                covariance,
                system,
            )
            if dynamics.raw_spectral_radius >= self.max_spectral_radius:
                unstable_rejections += 1
                continue
            retained.append(
                _PosteriorState(
                    coefficient_vector=coefficient_vector.copy(),
                    standardized_covariance=covariance.copy(),
                    dynamics=dynamics,
                )
            )

        self.gibbs_iterations_ = iteration
        self.unstable_rejections_ = unstable_rejections
        self.coefficient_draws_ = np.stack(
            [state.coefficient_vector for state in retained]
        )
        self.standardized_covariance_draws_ = np.stack(
            [state.standardized_covariance for state in retained]
        )
        self.posterior_mean_ = self.coefficient_draws_.mean(axis=0)
        self.posterior_covariance_mean_ = self.standardized_covariance_draws_.mean(
            axis=0
        )
        distances = np.sum(
            (self.coefficient_draws_ - self.posterior_mean_[None, :]) ** 2,
            axis=1,
        )
        self.representative_draw_index_ = int(np.argmin(distances))
        self.draws_ = [state.dynamics for state in retained]
        self.dynamics_ = self.draws_[self.representative_draw_index_]
        self.last_draw_assignments_ = None
        self.additional_posterior_unstable_rejections_ = 0
        self._continuation_covariance_ = covariance.copy()
        return self

    def draw_additional_posterior_dynamics(
        self,
        n_states: int,
        rng: np.random.Generator,
    ) -> list[LinearDynamics]:
        """Continue the fitted Gibbs chain and return extra stable states.

        The returned dynamics preserve the same paired coefficient/covariance
        target and raw-stability truncation as :meth:`fit`.  They do not alter
        ``draws_`` or any reported posterior summaries.  Only the private
        continuation state advances, which allows an adaptive conditional
        sampler to request successive batches without restarting burn-in.
        """
        n_states = int(n_states)
        if n_states < 1:
            raise ValueError("n_states must be positive")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        required = (
            self._fitted_system_,
            self._fitted_prior_mean_,
            self._fitted_prior_precision_,
            self._fitted_prior_df_,
            self._fitted_prior_scale_,
            self._continuation_covariance_,
        )
        if any(value is None for value in required):
            raise RuntimeError("generator not fitted")

        system = self._fitted_system_
        prior_mean = self._fitted_prior_mean_
        prior_precision = self._fitted_prior_precision_
        prior_df = float(self._fitted_prior_df_)
        prior_scale = self._fitted_prior_scale_
        covariance = self._continuation_covariance_.copy()
        retained: list[LinearDynamics] = []
        rejected = 0
        iterations = 0
        # The original cap is already deliberately generous for 100 states.
        # Scale it linearly for a larger requested enrichment batch.
        iteration_cap = max(
            self.max_sampling_iterations,
            int(np.ceil(n_states / self.n_dynamics)) * self.max_sampling_iterations,
        )
        while len(retained) < n_states:
            coefficient_vector = None
            for _ in range(self.thin):
                iterations += 1
                if iterations > iteration_cap:
                    self._continuation_covariance_ = covariance.copy()
                    self.additional_posterior_unstable_rejections_ += rejected
                    raise RuntimeError(
                        "insufficient stable continuation states; "
                        f"retained {len(retained)} of {n_states} after "
                        f"{iteration_cap} Gibbs iterations"
                    )
                coefficient_vector = self._draw_coefficients(
                    system,
                    covariance,
                    prior_mean,
                    prior_precision,
                    rng,
                )
                covariance = self._draw_covariance(
                    system,
                    coefficient_vector,
                    prior_df,
                    prior_scale,
                    rng,
                )
            dynamics = self._to_raw_dynamics(
                coefficient_vector,
                covariance,
                system,
            )
            if dynamics.raw_spectral_radius >= self.max_spectral_radius:
                rejected += 1
                continue
            retained.append(dynamics)

        self._continuation_covariance_ = covariance.copy()
        self.additional_posterior_unstable_rejections_ += rejected
        return retained

    # --------------------------------------------------------------- sample
    def _validate_sampling_inputs(
        self,
        n_paths: int,
        horizon: int,
        jumpoff: np.ndarray | None,
        anchor: np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if n_paths < 1 or horizon < 1:
            raise ValueError("n_paths and horizon must be positive")
        n_features = len(self.features)
        if anchor is None and jumpoff is None:
            raise RuntimeError("unanchored sampling needs a jumpoff state")
        checked_jumpoff = None
        if jumpoff is not None:
            checked_jumpoff = np.asarray(jumpoff, dtype=float)
            if checked_jumpoff.shape != (n_features,):
                raise ValueError("jumpoff shape must be (n_features,)")
            if not np.isfinite(checked_jumpoff).all():
                raise ValueError("jumpoff must contain only finite values")
        checked_anchor = None
        if anchor is not None:
            checked_anchor = np.asarray(anchor, dtype=float)
            if checked_anchor.shape != (horizon, n_features):
                raise ValueError("anchor shape must be (horizon, n_features)")
            if not np.isfinite(checked_anchor).all():
                raise ValueError("anchor must contain only finite values")
        return checked_jumpoff, checked_anchor

    @staticmethod
    def _draw_iid_gaussian_shocks(
        covariance: np.ndarray,
        n_paths: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        return rng.multivariate_normal(
            np.zeros(covariance.shape[0]),
            covariance,
            size=(n_paths, horizon),
        )

    def sample(
        self,
        n_paths: int,
        horizon: int,
        rng: np.random.Generator,
        jumpoff: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.draws_ is None:
            raise RuntimeError("generator not fitted")
        checked_jumpoff, checked_anchor = self._validate_sampling_inputs(
            n_paths,
            horizon,
            jumpoff,
            anchor,
        )
        assignments = rng.integers(0, len(self.draws_), size=n_paths)
        self.last_draw_assignments_ = assignments.copy()
        paths = np.empty((n_paths, horizon, len(self.features)), dtype=float)

        for draw_index in np.unique(assignments):
            selected = assignments == draw_index
            count = int(selected.sum())
            dynamics = self.draws_[int(draw_index)]
            shocks = self._draw_iid_gaussian_shocks(
                dynamics.residual_cov,
                count,
                horizon,
                rng,
            )
            if checked_anchor is not None:
                deviation = np.zeros((count, len(self.features)), dtype=float)
                for step in range(horizon):
                    deviation = deviation @ dynamics.coef.T + shocks[:, step]
                    paths[selected, step] = checked_anchor[step][None, :] + deviation
            else:
                state = np.tile(checked_jumpoff[None, :], (count, 1))
                for step in range(horizon):
                    state = (
                        state @ dynamics.coef.T
                        + dynamics.intercept[None, :]
                        + shocks[:, step]
                    )
                    paths[selected, step] = state
        return clip_to_bounds(paths, self.features, self.bounds)

    # ------------------------------------------------------------ likelihood
    @staticmethod
    def _draw_loglik(
        dynamics: LinearDynamics,
        path: np.ndarray,
        jumpoff: np.ndarray,
    ) -> float:
        covariance = _symmetric_positive_definite(dynamics.residual_cov)
        sign, log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise RuntimeError("innovation covariance is not positive definite")
        precision = np.linalg.inv(covariance)
        previous = np.asarray(jumpoff, dtype=float)
        quadratic = 0.0
        for value in path:
            innovation = value - (dynamics.coef @ previous + dynamics.intercept)
            quadratic += float(innovation @ precision @ innovation)
            previous = value
        dimension = path.shape[0] * path.shape[1]
        return float(
            -0.5
            * (
                quadratic
                + path.shape[0] * log_determinant
                + dimension * np.log(2.0 * np.pi)
            )
        )

    def innovation_loglik(self, path: np.ndarray, jumpoff: np.ndarray) -> float:
        """Monte Carlo Gaussian posterior-predictive innovation log density."""
        if self.draws_ is None:
            raise RuntimeError("generator not fitted")
        path = np.asarray(path, dtype=float)
        jumpoff = np.asarray(jumpoff, dtype=float)
        if path.ndim != 2 or path.shape[1] != len(self.features):
            raise ValueError("path shape must be (horizon, n_features)")
        if jumpoff.shape != (len(self.features),):
            raise ValueError("jumpoff shape must be (n_features,)")
        if not np.isfinite(path).all() or not np.isfinite(jumpoff).all():
            raise ValueError("path and jumpoff must contain only finite values")
        values = np.asarray(
            [self._draw_loglik(draw, path, jumpoff) for draw in self.draws_],
            dtype=float,
        )
        return float(logsumexp(values) - np.log(len(values)))

    # ----------------------------------------------------------- diagnostics
    def sampler_diagnostics(self) -> pd.DataFrame:
        """Summarise stability filtering and lightweight chain dependence."""
        if self.draws_ is None or self.standardized_covariance_draws_ is None:
            raise RuntimeError("generator not fitted")
        radii = np.asarray(
            [draw.raw_spectral_radius for draw in self.draws_],
            dtype=float,
        )
        mean_variances = np.trace(
            self.standardized_covariance_draws_,
            axis1=1,
            axis2=2,
        ) / len(self.features)
        radius_rho, radius_ess = _lag1_diagnostics(radii)
        variance_rho, variance_ess = _lag1_diagnostics(mean_variances)
        candidates = len(self.draws_) + self.unstable_rejections_
        return pd.DataFrame(
            [
                {
                    "generator": self.name,
                    "retained_draws": len(self.draws_),
                    "gibbs_iterations": self.gibbs_iterations_,
                    "burn_in": self.burn_in,
                    "thin": self.thin,
                    "max_sampling_iterations": self.max_sampling_iterations,
                    "unstable_rejections": self.unstable_rejections_,
                    "stable_acceptance_rate": len(self.draws_) / candidates,
                    "min_spectral_radius": float(radii.min()),
                    "max_spectral_radius": float(radii.max()),
                    "spectral_radius_lag1_autocorrelation": radius_rho,
                    "spectral_radius_lag1_ess": radius_ess,
                    "mean_innovation_variance_lag1_autocorrelation": variance_rho,
                    "mean_innovation_variance_lag1_ess": variance_ess,
                    "representative_draw_index": self.representative_draw_index_,
                }
            ]
        )

    def stability_diagnostics(self) -> pd.DataFrame:
        """Compatibility view of the sampler's stability diagnostics."""
        return self.sampler_diagnostics()


class MinnesotaPosteriorVARGenerator(FullPosteriorVARGenerator):
    """Full posterior-predictive Minnesota BVAR benchmark."""

    name = FULL_BVAR_NAME

    def __init__(
        self,
        n_dynamics: int = 100,
        delta: float = 0.9,
        own_scale: float = 0.2,
        cross_scale: float = 0.1,
        intercept_scale: float = 10.0,
        covariance_prior_df: float | None = None,
        burn_in: int = 150,
        thin: int = 3,
        max_spectral_radius: float = 0.995,
        max_sampling_iterations: int = 20000,
        seed: int = 211,
        *,
        features: Sequence[str] | None = None,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        own_lag_means: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__(
            n_dynamics=n_dynamics,
            delta=delta,
            own_scale=own_scale,
            cross_scale=cross_scale,
            intercept_scale=intercept_scale,
            covariance_prior_df=covariance_prior_df,
            burn_in=burn_in,
            thin=thin,
            max_spectral_radius=max_spectral_radius,
            max_sampling_iterations=max_sampling_iterations,
            seed=seed,
            expert_weight=None,
            features=features,
            bounds=bounds,
            own_lag_means=own_lag_means,
        )


class SoftStructuralPosteriorVARGenerator(FullPosteriorVARGenerator):
    """Full posterior BVAR with a soft expert-informed coefficient prior."""

    name = FULL_SSP_NAME

    def __init__(
        self,
        n_dynamics: int = 100,
        delta: float = 0.9,
        own_scale: float = 0.2,
        cross_scale: float = 0.1,
        expert_weight: float = 0.35,
        intercept_scale: float = 10.0,
        covariance_prior_df: float | None = None,
        burn_in: int = 150,
        thin: int = 3,
        max_spectral_radius: float = 0.995,
        max_sampling_iterations: int = 20000,
        seed: int = 211,
        *,
        features: Sequence[str] | None = None,
        expert_edges: Mapping[str, Sequence[str]] | None = None,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        own_lag_means: Mapping[str, float] | None = None,
    ) -> None:
        super().__init__(
            n_dynamics=n_dynamics,
            delta=delta,
            own_scale=own_scale,
            cross_scale=cross_scale,
            intercept_scale=intercept_scale,
            covariance_prior_df=covariance_prior_df,
            burn_in=burn_in,
            thin=thin,
            max_spectral_radius=max_spectral_radius,
            max_sampling_iterations=max_sampling_iterations,
            seed=seed,
            expert_weight=expert_weight,
            features=features,
            expert_edges=expert_edges,
            bounds=bounds,
            own_lag_means=own_lag_means,
        )


__all__ = [
    "FULL_BVAR_NAME",
    "FULL_SSP_NAME",
    "FullPosteriorVARGenerator",
    "MinnesotaPosteriorVARGenerator",
    "SoftStructuralPosteriorVARGenerator",
]
