"""Independent univariate AR-t benchmark for the fair rolling evaluation.

Each feature is modelled separately.  AR(1) and AR(2) candidates jointly
estimate an intercept, stationary autoregressive coefficients, Student-t
scale, and Student-t degrees of freedom by conditional maximum likelihood.
The two lag orders are compared using BIC on the same observations, beginning
after the maximum candidate lag.  This keeps lag selection invariant to the
units in which a series is measured.

The benchmark deliberately contains no cross-variable dynamics.  Conditional
completion draws the ordinary marginal forecasts and then restores only the
supplied feature path.  Thus, with matched random numbers, every hidden path
is exactly invariant to conditioning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln

MAX_AR_ORDER = 2
DEFAULT_MAX_SPECTRAL_RADIUS = 0.995
MINIMUM_INNOVATION_SCALE = float(np.nextafter(0.001, np.inf))
DEFAULT_MIN_DEGREES_OF_FREEDOM = 2.1
DEFAULT_MAX_DEGREES_OF_FREEDOM = 30.0
_PACF_PARAMETER_BOUND = 6.0


@dataclass(frozen=True)
class UnivariateARFit:
    """One selected or candidate univariate AR-t maximum-likelihood fit."""

    order: int
    coefficients: np.ndarray
    intercept: float
    scale: float
    degrees_of_freedom: float
    residuals: np.ndarray
    log_likelihood: float
    bic: float
    spectral_radius: float
    raw_ols_spectral_radius: float
    converged: bool
    optimizer_status: int
    optimizer_message: str
    optimizer_iterations: int


@dataclass(frozen=True)
class FailedUnivariateARFit:
    """Auditable record for a candidate whose joint likelihood did not converge."""

    order: int
    raw_ols_spectral_radius: float
    optimizer_status: int
    optimizer_message: str
    optimizer_iterations: int


@dataclass(frozen=True)
class IndependentARDynamics:
    """Selected independent AR equations in an engine-friendly array form."""

    features: tuple[str, ...]
    orders: np.ndarray
    coef: np.ndarray
    intercept: np.ndarray
    residuals: np.ndarray
    residual_cov: np.ndarray
    candidate_bics: np.ndarray
    candidate_converged: np.ndarray
    raw_ols_spectral_radii: np.ndarray
    spectral_radii: np.ndarray
    innovation_df: np.ndarray
    innovation_scale: np.ndarray
    log_likelihoods: np.ndarray
    optimizer_status: np.ndarray
    optimizer_iterations: np.ndarray
    training_rows: int

    @property
    def raw_spectral_radius(self) -> float:
        return float(np.max(self.raw_ols_spectral_radii))

    @property
    def spectral_radius(self) -> float:
        return float(np.max(self.spectral_radii))


@dataclass(frozen=True)
class IndependentARConditionalDiagnostics:
    """Diagnostics identifying the intentional no-cross-update conditioner."""

    diagnostic_type: str
    observed_features: tuple[str, ...]
    cross_variable_updating: bool
    hidden_paths_unmodified: bool
    maximum_observed_restoration_error: float


def _spectral_radius(coefficients: np.ndarray) -> float:
    """Return the companion-matrix spectral radius for an AR(1) or AR(2)."""
    phi = np.asarray(coefficients, dtype=float)
    if phi.ndim != 1 or phi.size not in (1, 2):
        raise ValueError("AR coefficients must contain one or two values")
    if phi.size == 1:
        return float(abs(phi[0]))
    companion = np.asarray([[phi[0], phi[1]], [1.0, 0.0]], dtype=float)
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def _stationary_projection(
    coefficients: np.ndarray,
    max_spectral_radius: float,
) -> np.ndarray:
    """Provide a stable OLS initializer without defining the fitted model."""
    phi = np.asarray(coefficients, dtype=float)
    if _spectral_radius(phi) < max_spectral_radius:
        return phi.copy()
    lower = 0.0
    upper = 1.0
    for _ in range(80):
        scale = 0.5 * (lower + upper)
        if _spectral_radius(scale * phi) < max_spectral_radius:
            lower = scale
        else:
            upper = scale
    return lower * phi


def _coefficients_from_pacf_parameters(
    parameters: np.ndarray,
    order: int,
    max_spectral_radius: float,
) -> np.ndarray:
    """Map unconstrained partial autocorrelations to a capped stable AR."""
    partial = np.tanh(np.asarray(parameters[:order], dtype=float))
    if order == 1:
        normalized = np.asarray([partial[0]], dtype=float)
    elif order == 2:
        # Durbin-Levinson recursion: phi_2,1=kappa_1(1-kappa_2),
        # phi_2,2=kappa_2.  |kappa_j|<1 is exactly the AR(2) stable region.
        normalized = np.asarray(
            [partial[0] * (1.0 - partial[1]), partial[1]],
            dtype=float,
        )
    else:
        raise ValueError("only AR(1) and AR(2) candidates are supported")
    powers = max_spectral_radius ** np.arange(1, order + 1, dtype=float)
    return normalized * powers


def _pacf_parameters_from_coefficients(
    coefficients: np.ndarray,
    max_spectral_radius: float,
) -> np.ndarray:
    """Invert the AR(1)/AR(2) transform for a stable OLS initializer."""
    phi = np.asarray(coefficients, dtype=float)
    if phi.size == 1:
        partial = np.asarray([phi[0] / max_spectral_radius])
    else:
        second = phi[1] / max_spectral_radius**2
        first = (phi[0] / max_spectral_radius) / max(1.0 - second, 1e-8)
        partial = np.asarray([first, second])
    partial = np.clip(partial, -0.995, 0.995)
    return np.arctanh(partial)


def _student_t_log_likelihood(
    residuals: np.ndarray,
    scale: float,
    degrees_of_freedom: float,
) -> float:
    """Conditional log likelihood for zero-location Student-t innovations."""
    n_observations = len(residuals)
    constant = (
        gammaln((degrees_of_freedom + 1.0) / 2.0)
        - gammaln(degrees_of_freedom / 2.0)
        - 0.5 * np.log(degrees_of_freedom * np.pi)
        - np.log(scale)
    )
    standardized_square = (residuals / scale) ** 2
    return float(
        n_observations * constant
        - 0.5
        * (degrees_of_freedom + 1.0)
        * np.sum(np.log1p(standardized_square / degrees_of_freedom))
    )


def _fit_candidate(
    values: np.ndarray,
    order: int,
    max_spectral_radius: float,
    minimum_df: float,
    maximum_df: float,
    *,
    hold_back: int = MAX_AR_ORDER,
) -> UnivariateARFit | FailedUnivariateARFit:
    """Jointly fit one stationary AR-t candidate by conditional MLE."""
    if order not in (1, 2):
        raise ValueError("only AR(1) and AR(2) candidates are supported")
    if hold_back < order or hold_back >= len(values):
        raise ValueError("hold_back must cover the candidate lag and leave data")
    target_raw = np.asarray(values[hold_back:], dtype=float)
    lagged_raw = np.column_stack(
        [
            values[hold_back - lag : len(values) - lag]
            for lag in range(1, order + 1)
        ]
    )
    center = float(np.mean(values))
    normalizer = float(np.std(values, ddof=0))
    if normalizer <= 1e-8:
        normalizer = 1.0
    target = (target_raw - center) / normalizer
    lagged = (lagged_raw - center) / normalizer

    design = np.column_stack([np.ones(len(target)), lagged])
    ols, *_ = np.linalg.lstsq(design, target, rcond=None)
    raw_ols_radius = _spectral_radius(ols[1:])
    initial_coefficients = _stationary_projection(
        ols[1:],
        max_spectral_radius * (1.0 - 1e-8),
    )
    initial_pacf = _pacf_parameters_from_coefficients(
        initial_coefficients,
        max_spectral_radius,
    )
    initial_intercept = float(np.mean(target - lagged @ initial_coefficients))
    initial_residuals_raw = normalizer * (
        target - initial_intercept - lagged @ initial_coefficients
    )
    initial_df = min(max(5.0, minimum_df), maximum_df)
    initial_rms = float(np.sqrt(np.mean(initial_residuals_raw**2)))
    initial_scale = max(
        initial_rms * np.sqrt((initial_df - 2.0) / initial_df),
        MINIMUM_INNOVATION_SCALE * 1.05,
    )
    initial = np.concatenate(
        [
            np.asarray([initial_intercept]),
            initial_pacf,
            np.asarray([np.log(initial_scale), initial_df]),
        ]
    )
    scale_upper = max(
        MINIMUM_INNOVATION_SCALE * 100.0,
        float(np.ptp(values)) * 100.0,
        normalizer * 100.0,
    )
    bounds = (
        [(None, None)]
        + [(-_PACF_PARAMETER_BOUND, _PACF_PARAMETER_BOUND)] * order
        + [
            (np.log(MINIMUM_INNOVATION_SCALE), np.log(scale_upper)),
            (minimum_df, maximum_df),
        ]
    )

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray]:
        coefficients = _coefficients_from_pacf_parameters(
            parameters[1 : 1 + order],
            order,
            max_spectral_radius,
        )
        scale = float(np.exp(parameters[-2]))
        degrees_of_freedom = float(parameters[-1])
        residuals = normalizer * (target - parameters[0] - lagged @ coefficients)
        return coefficients, scale, degrees_of_freedom, residuals

    def objective(parameters: np.ndarray) -> float:
        _, scale, degrees_of_freedom, residuals = unpack(parameters)
        log_likelihood = _student_t_log_likelihood(
            residuals,
            scale,
            degrees_of_freedom,
        )
        return float(-log_likelihood) if np.isfinite(log_likelihood) else 1e100

    attempts = []
    for starting_df in dict.fromkeys((initial_df, minimum_df + 0.5, maximum_df)):
        start = initial.copy()
        start[-1] = float(np.clip(starting_df, minimum_df, maximum_df))
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={
                "ftol": 1e-10,
                "gtol": 1e-6,
                "maxiter": 500,
                "maxls": 50,
            },
        )
        attempts.append(result)
    converged = [
        result
        for result in attempts
        if bool(result.success)
        and np.isfinite(result.fun)
        and np.isfinite(result.x).all()
    ]
    if not converged:
        best = min(attempts, key=lambda result: float(result.fun))
        return FailedUnivariateARFit(
            order=order,
            raw_ols_spectral_radius=raw_ols_radius,
            optimizer_status=int(best.status),
            optimizer_message=str(best.message),
            optimizer_iterations=int(getattr(best, "nit", 0)),
        )

    result = min(converged, key=lambda candidate: float(candidate.fun))
    coefficients, scale, degrees_of_freedom, residuals = unpack(result.x)
    intercept_standardized = float(result.x[0])
    intercept = (
        center * (1.0 - float(np.sum(coefficients)))
        + normalizer * intercept_standardized
    )
    log_likelihood = _student_t_log_likelihood(
        residuals,
        scale,
        degrees_of_freedom,
    )
    n_observations = len(target_raw)
    parameter_count = order + 3
    bic = -2.0 * log_likelihood + parameter_count * np.log(n_observations)
    radius = _spectral_radius(coefficients)
    if not radius < max_spectral_radius:
        raise RuntimeError(
            "stationary AR transform produced an invalid coefficient fit"
        )
    return UnivariateARFit(
        order=order,
        coefficients=coefficients,
        intercept=float(intercept),
        scale=scale,
        degrees_of_freedom=degrees_of_freedom,
        residuals=residuals,
        log_likelihood=log_likelihood,
        bic=float(bic),
        spectral_radius=radius,
        raw_ols_spectral_radius=raw_ols_radius,
        converged=True,
        optimizer_status=int(result.status),
        optimizer_message=str(result.message),
        optimizer_iterations=int(getattr(result, "nit", 0)),
    )


class UnivariateARGenerator:
    """Independent AR(1)/AR(2) benchmark with Student-t innovations."""

    name = "univariate_ar_t"

    def __init__(
        self,
        *,
        features: Sequence[str] | None = None,
        bounds: Mapping[str, tuple[float, float]] | None = None,
        max_spectral_radius: float = DEFAULT_MAX_SPECTRAL_RADIUS,
        minimum_degrees_of_freedom: float = DEFAULT_MIN_DEGREES_OF_FREEDOM,
        maximum_degrees_of_freedom: float = DEFAULT_MAX_DEGREES_OF_FREEDOM,
    ) -> None:
        configured_features = None if features is None else tuple(features)
        if configured_features is not None and (
            not configured_features
            or len(set(configured_features)) != len(configured_features)
        ):
            raise ValueError("features must be non-empty and unique")
        if not 0.0 < max_spectral_radius < 1.0:
            raise ValueError(
                "max_spectral_radius must lie strictly between zero and one"
            )
        if minimum_degrees_of_freedom < DEFAULT_MIN_DEGREES_OF_FREEDOM:
            raise ValueError("minimum_degrees_of_freedom must be at least 2.1")
        if maximum_degrees_of_freedom > DEFAULT_MAX_DEGREES_OF_FREEDOM:
            raise ValueError("maximum_degrees_of_freedom must not exceed 30")
        if maximum_degrees_of_freedom < minimum_degrees_of_freedom:
            raise ValueError("maximum_degrees_of_freedom must not be below the minimum")
        self._configured_features = configured_features
        self.features = configured_features or ()
        self.bounds = None if bounds is None else dict(bounds)
        self.max_spectral_radius = float(max_spectral_radius)
        self.minimum_degrees_of_freedom = float(minimum_degrees_of_freedom)
        self.maximum_degrees_of_freedom = float(maximum_degrees_of_freedom)
        self.dynamics_: IndependentARDynamics | None = None
        self.training_tail_: np.ndarray | None = None
        self.fits_: tuple[UnivariateARFit, ...] = ()
        self.candidate_fits_: tuple[
            tuple[UnivariateARFit | FailedUnivariateARFit, ...], ...
        ] = ()

    def fit(self, history: pd.DataFrame) -> UnivariateARGenerator:
        """Fit every equation using only the supplied chronological history."""
        if not isinstance(history, pd.DataFrame):
            raise TypeError("history must be a pandas DataFrame")
        features = self._configured_features or tuple(
            str(name) for name in history.columns
        )
        missing = [feature for feature in features if feature not in history.columns]
        if missing:
            raise ValueError(f"history is missing configured features: {missing}")
        if len(history) < 8:
            raise ValueError("at least eight training rows are required")
        values = history[list(features)].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("history must contain only finite values")

        selected: list[UnivariateARFit] = []
        all_candidates: list[tuple[UnivariateARFit | FailedUnivariateARFit, ...]] = []
        candidate_bics = np.full((len(features), MAX_AR_ORDER), np.inf, dtype=float)
        candidate_converged = np.zeros(
            (len(features), MAX_AR_ORDER),
            dtype=bool,
        )
        for feature_index, feature in enumerate(features):
            candidates = tuple(
                _fit_candidate(
                    values[:, feature_index],
                    order,
                    self.max_spectral_radius,
                    self.minimum_degrees_of_freedom,
                    self.maximum_degrees_of_freedom,
                )
                for order in (1, 2)
            )
            all_candidates.append(candidates)
            successful = []
            for candidate in candidates:
                if isinstance(candidate, UnivariateARFit):
                    candidate_bics[feature_index, candidate.order - 1] = candidate.bic
                    candidate_converged[feature_index, candidate.order - 1] = True
                    successful.append(candidate)
            if not successful:
                messages = "; ".join(
                    f"AR({candidate.order}): {candidate.optimizer_message}"
                    for candidate in candidates
                )
                raise RuntimeError(
                    f"joint AR-t likelihood failed for feature {feature!r}: {messages}"
                )
            selected.append(
                min(successful, key=lambda candidate: (candidate.bic, candidate.order))
            )

        coefficients = np.zeros((len(features), MAX_AR_ORDER), dtype=float)
        max_residual_rows = len(history) - MAX_AR_ORDER
        residuals = np.empty((max_residual_rows, len(features)), dtype=float)
        for feature_index, fitted in enumerate(selected):
            coefficients[feature_index, : fitted.order] = fitted.coefficients
            residuals[:, feature_index] = fitted.residuals[-max_residual_rows:]
        innovation_df = np.asarray(
            [fitted.degrees_of_freedom for fitted in selected],
            dtype=float,
        )
        innovation_scale = np.asarray(
            [fitted.scale for fitted in selected],
            dtype=float,
        )
        residual_variance = innovation_scale**2 * innovation_df / (innovation_df - 2.0)
        self.features = features
        self.training_tail_ = values[-MAX_AR_ORDER:].copy()
        self.fits_ = tuple(selected)
        self.candidate_fits_ = tuple(all_candidates)
        self.dynamics_ = IndependentARDynamics(
            features=features,
            orders=np.asarray([fit.order for fit in selected], dtype=int),
            coef=coefficients,
            intercept=np.asarray([fit.intercept for fit in selected], dtype=float),
            residuals=residuals,
            residual_cov=np.diag(residual_variance),
            candidate_bics=candidate_bics,
            candidate_converged=candidate_converged,
            raw_ols_spectral_radii=np.asarray(
                [fit.raw_ols_spectral_radius for fit in selected],
                dtype=float,
            ),
            spectral_radii=np.asarray(
                [fit.spectral_radius for fit in selected],
                dtype=float,
            ),
            innovation_df=innovation_df,
            innovation_scale=innovation_scale,
            log_likelihoods=np.asarray(
                [fit.log_likelihood for fit in selected],
                dtype=float,
            ),
            optimizer_status=np.asarray(
                [fit.optimizer_status for fit in selected],
                dtype=int,
            ),
            optimizer_iterations=np.asarray(
                [fit.optimizer_iterations for fit in selected],
                dtype=int,
            ),
            training_rows=len(history),
        )
        return self

    def _initial_tail(self, jumpoff: np.ndarray | None) -> np.ndarray:
        if self.dynamics_ is None or self.training_tail_ is None:
            raise RuntimeError("generator not fitted")
        tail = self.training_tail_.copy()
        if jumpoff is None:
            return tail
        supplied = np.asarray(jumpoff, dtype=float)
        n_features = len(self.features)
        if supplied.shape == (n_features,):
            tail[-1] = supplied
        elif supplied.shape == (MAX_AR_ORDER, n_features):
            tail = supplied.copy()
        else:
            raise ValueError(
                "jumpoff must match the feature vector or the final two states"
            )
        if not np.isfinite(tail).all():
            raise ValueError("jumpoff must contain only finite values")
        return tail

    def _apply_bounds(self, paths: np.ndarray) -> np.ndarray:
        if self.bounds is None:
            return paths
        bounded = paths.copy()
        for feature_index, feature in enumerate(self.features):
            if feature in self.bounds:
                lower, upper = self.bounds[feature]
                bounded[..., feature_index] = np.clip(
                    bounded[..., feature_index],
                    lower,
                    upper,
                )
        return bounded

    def sample(
        self,
        n_paths: int,
        horizon: int,
        rng: np.random.Generator,
        jumpoff: np.ndarray | None = None,
        *,
        apply_bounds: bool = False,
    ) -> np.ndarray:
        """Draw a path cube with shape ``(n_paths, horizon, n_features)``."""
        if self.dynamics_ is None:
            raise RuntimeError("generator not fitted")
        if n_paths < 1 or horizon < 1:
            raise ValueError("n_paths and horizon must be positive")
        tail = self._initial_tail(jumpoff)
        dynamics = self.dynamics_
        paths = np.empty((n_paths, horizon, len(self.features)), dtype=float)
        for feature_index in range(len(self.features)):
            previous = np.full(n_paths, tail[-1, feature_index], dtype=float)
            previous_two = np.full(n_paths, tail[-2, feature_index], dtype=float)
            shocks = (
                rng.standard_t(
                    dynamics.innovation_df[feature_index],
                    size=(n_paths, horizon),
                )
                * dynamics.innovation_scale[feature_index]
            )
            for step in range(horizon):
                state = (
                    dynamics.intercept[feature_index]
                    + dynamics.coef[feature_index, 0] * previous
                    + dynamics.coef[feature_index, 1] * previous_two
                    + shocks[:, step]
                )
                paths[:, step, feature_index] = state
                previous_two, previous = previous, state
        return self._apply_bounds(paths) if apply_bounds else paths

    def sample_conditional(
        self,
        supplied_path: np.ndarray,
        observed_features: Sequence[str],
        n_paths: int,
        rng: np.random.Generator,
        jumpoff: np.ndarray | None = None,
        *,
        apply_hidden_bounds: bool = False,
    ) -> tuple[np.ndarray, IndependentARConditionalDiagnostics]:
        """Complete a supplied path without updating any other variable."""
        if self.dynamics_ is None:
            raise RuntimeError("generator not fitted")
        observed = tuple(observed_features)
        if not observed:
            raise ValueError("observe at least one feature")
        unknown = [feature for feature in observed if feature not in self.features]
        if unknown:
            raise ValueError(f"unknown observed features: {unknown}")
        if len(set(observed)) != len(observed):
            raise ValueError("observed_features must not contain duplicates")
        if len(observed) == len(self.features):
            raise ValueError("cannot observe every feature; none would remain to score")
        supplied = np.asarray(supplied_path, dtype=float)
        if supplied.ndim != 2 or supplied.shape[1] != len(self.features):
            raise ValueError("supplied_path must have shape (horizon, len(features))")
        if supplied.shape[0] < 1 or not np.isfinite(supplied).all():
            raise ValueError("supplied_path must be non-empty and finite")

        paths = self.sample(
            n_paths,
            supplied.shape[0],
            rng,
            jumpoff=jumpoff,
            apply_bounds=apply_hidden_bounds,
        )
        for feature in observed:
            feature_index = self.features.index(feature)
            paths[:, :, feature_index] = supplied[None, :, feature_index]
        restoration_error = max(
            float(
                np.max(
                    np.abs(
                        paths[:, :, self.features.index(feature)]
                        - supplied[None, :, self.features.index(feature)]
                    )
                )
            )
            for feature in observed
        )
        diagnostics = IndependentARConditionalDiagnostics(
            diagnostic_type="independent_univariate",
            observed_features=observed,
            cross_variable_updating=False,
            hidden_paths_unmodified=True,
            maximum_observed_restoration_error=restoration_error,
        )
        return paths, diagnostics

    def fingerprint_arrays(self) -> dict[str, np.ndarray]:
        """Expose every selected numerical law component for provenance hashes."""
        if self.dynamics_ is None or self.training_tail_ is None:
            raise RuntimeError("generator not fitted")
        dynamics = self.dynamics_
        return {
            "orders": dynamics.orders,
            "coef": dynamics.coef,
            "intercept": dynamics.intercept,
            "innovation_df": dynamics.innovation_df,
            "innovation_scale": dynamics.innovation_scale,
            "candidate_bics": dynamics.candidate_bics,
            "candidate_converged": dynamics.candidate_converged,
            "spectral_radii": dynamics.spectral_radii,
            "log_likelihoods": dynamics.log_likelihoods,
            "training_tail": self.training_tail_,
        }

    def diagnostics(self) -> pd.DataFrame:
        """Return one deterministic selected-equation row per feature."""
        if self.dynamics_ is None:
            raise RuntimeError("generator not fitted")
        rows = []
        for feature, fitted, candidates in zip(
            self.features,
            self.fits_,
            self.candidate_fits_,
            strict=True,
        ):
            candidate_status = {candidate.order: candidate for candidate in candidates}
            rows.append(
                {
                    "feature": feature,
                    "selected_lag": int(fitted.order),
                    "bic": float(fitted.bic),
                    "log_likelihood": float(fitted.log_likelihood),
                    "intercept": float(fitted.intercept),
                    "ar_lag_1": float(fitted.coefficients[0]),
                    "ar_lag_2": (
                        float(fitted.coefficients[1]) if fitted.order == 2 else 0.0
                    ),
                    "scale": float(fitted.scale),
                    "df": float(fitted.degrees_of_freedom),
                    "converged": bool(fitted.converged),
                    "optimizer_status": int(fitted.optimizer_status),
                    "optimizer_iterations": int(fitted.optimizer_iterations),
                    "raw_ols_spectral_radius": float(fitted.raw_ols_spectral_radius),
                    "max_companion_spectral_radius": float(fitted.spectral_radius),
                    "stationarity_margin": float(1.0 - fitted.spectral_radius),
                    "bic_ar1": (
                        float(candidate_status[1].bic)
                        if isinstance(candidate_status[1], UnivariateARFit)
                        else np.inf
                    ),
                    "bic_ar2": (
                        float(candidate_status[2].bic)
                        if isinstance(candidate_status[2], UnivariateARFit)
                        else np.inf
                    ),
                    "ar1_converged": isinstance(
                        candidate_status[1],
                        UnivariateARFit,
                    ),
                    "ar2_converged": isinstance(
                        candidate_status[2],
                        UnivariateARFit,
                    ),
                    "ar1_optimizer_status": int(candidate_status[1].optimizer_status),
                    "ar2_optimizer_status": int(candidate_status[2].optimizer_status),
                    "ar1_optimizer_message": str(candidate_status[1].optimizer_message),
                    "ar2_optimizer_message": str(candidate_status[2].optimizer_message),
                    "training_rows": int(self.dynamics_.training_rows),
                }
            )
        return pd.DataFrame(rows)


__all__ = [
    "FailedUnivariateARFit",
    "IndependentARConditionalDiagnostics",
    "IndependentARDynamics",
    "UnivariateARFit",
    "UnivariateARGenerator",
]
