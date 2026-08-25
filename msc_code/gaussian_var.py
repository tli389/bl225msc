"""Path generator family with independent structure and shock axes.

The graph-block generator differs from the VAR incumbent through sparse
coefficients and historically resampled, persistent multivariate shocks.
Keeping structure and shock choices independent gives controlled ablations

    coef_mode  in {dense, sparse}   x   shock_mode in {gaussian, bootstrap}

so the envelope backtest can attribute gains to structure, marginal shock
shape and temporal shock persistence. ``dense + gaussian`` is the VAR
incumbent and ``sparse + block`` is the proposed graph-structured generator.
A delta-bootstrap random walk with no dynamics is the structureless floor.

All generators share one interface:

    generator.fit(history_frame)                      # columns = FEATURES
    generator.sample(n_paths, horizon, rng)           # unanchored rollout
    generator.sample(..., anchor=path)                # deviations around an
                                                      # externally given path
returning an array of shape (n_paths, horizon, k).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression, lars_path, lasso_path
from sklearn.model_selection import TimeSeriesSplit

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

MAX_PARENTS = 3
MAX_SPECTRAL_RADIUS = 0.995


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


# Compatibility name retained for the frozen protocol modules.
_clip_to_bounds = clip_to_bounds


@dataclass
class LinearDynamics:
    """Fitted lag-1 linear system y_t = c + A y_{t-1} + e_t."""

    coef: np.ndarray  # (k, k), row = target equation
    intercept: np.ndarray  # (k,)
    residuals: np.ndarray  # (n, k)
    residual_cov: np.ndarray  # (k, k)
    raw_spectral_radius: float = np.nan
    stability_scale: float = 1.0


def stabilize_lag_matrix(
    coef: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    max_radius: float = MAX_SPECTRAL_RADIUS,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Project a lag-1 system into the stable region without changing support.

    For a VAR(1), stability requires the spectral radius of ``A`` to be below
    one. If needed, all lag coefficients are scaled by the same factor and the
    intercept is re-centred so the fitted unconditional sample mean is
    preserved. Edge signs and zeros are unchanged.
    """
    raw_radius = float(np.max(np.abs(np.linalg.eigvals(coef))))
    scale = 1.0 if raw_radius <= max_radius else max_radius / raw_radius
    stable = coef * scale
    intercept = y.mean(axis=0) - x.mean(axis=0) @ stable.T
    return stable, intercept, raw_radius, float(scale)


def _time_series_cv(n_rows: int) -> TimeSeriesSplit:
    """Forward-only folds for regularisation selection."""
    if n_rows < 12:
        raise ValueError("at least 12 transition rows are required")
    n_splits = min(5, max(2, n_rows // 12))
    return TimeSeriesSplit(n_splits=n_splits)


@dataclass
class _PartialledDesign:
    """Fold-fitted transformation for one sparse transition equation."""

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


def _partialled_design(
    x: np.ndarray,
    target_values: np.ndarray,
    target: int,
    penalty_weights: np.ndarray,
) -> _PartialledDesign:
    """Fit preprocessing using only the supplied rows."""
    n_rows, n_features = x.shape
    cross = np.asarray([i for i in range(n_features) if i != target], dtype=int)
    x_mean = x.mean(axis=0)
    x_scale = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
    xz = (x - x_mean) / x_scale
    own_design = np.column_stack([np.ones(n_rows), xz[:, target]])

    y_projection, *_ = np.linalg.lstsq(own_design, target_values, rcond=None)
    target_residual = target_values - own_design @ y_projection
    cross_values = xz[:, cross]
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
    """Apply a training-fold transformation to later rows."""
    xz = (x - fitted.x_mean) / fitted.x_scale
    own_design = np.column_stack([np.ones(len(x)), xz[:, fitted.target]])
    baseline = own_design @ fitted.y_projection
    cross_values = xz[:, fitted.cross]
    residual = cross_values - own_design @ fitted.x_projection
    design = residual / fitted.residual_scale / fitted.penalty_weights[None, :]
    return design, baseline


def _alpha_ratio_fold_local(
    x: np.ndarray,
    target_values: np.ndarray,
    target: int,
    penalty_weights: np.ndarray,
    path_solver: str = "coordinate",
) -> float:
    """Select a scale-free Lasso penalty with fully fold-local preprocessing."""
    if path_solver not in {"coordinate", "lars"}:
        raise ValueError("path_solver must be 'coordinate' or 'lars'")
    # Match sklearn's established LassoCV path width.  Weaker ratios become
    # numerically indistinguishable from an unregularised fit in these short,
    # collinear macro samples and can fail the coordinate-descent tolerance.
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
                # With only K-1 cross lags, LARS evaluates the same Lasso
                # objective without the convergence failures that coordinate
                # descent can encounter in highly collinear wider panels.
                path_alphas, _, path_coefficients = lars_path(
                    fitted.design,
                    fitted.target_residual,
                    method="lasso",
                    alpha_min=0.0,
                )
                order = np.argsort(path_alphas)
                unique_alphas, indices = np.unique(
                    path_alphas[order],
                    return_index=True,
                )
                ordered_coefficients = path_coefficients[:, order][:, indices]
                coefficients = np.vstack(
                    [
                        np.interp(alphas, unique_alphas, coefficient_path)
                        for coefficient_path in ordered_coefficients
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
    seed: int,
    penalty_weights: np.ndarray | None = None,
    path_solver: str = "coordinate",
) -> tuple[np.ndarray, float]:
    """Fit one sparse equation with an unpenalised own lag.

    Scaling and own-lag partialling are estimated separately inside each
    forward-validation fold.  ``penalty_weights`` implements weighted Lasso:
    values below one make supported cross-lags easier, but never mandatory.
    """
    del seed  # Coordinate descent and the forward folds are deterministic.
    n_rows, n_features = x.shape
    cross = np.asarray([i for i in range(n_features) if i != target], dtype=int)
    if penalty_weights is None:
        penalty_weights = np.ones(len(cross), dtype=float)
    penalty_weights = np.asarray(penalty_weights, dtype=float)
    if penalty_weights.shape != (len(cross),) or np.any(penalty_weights <= 0):
        raise ValueError("penalty_weights must be positive and match cross-lags")

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
    xz = (x - fitted.x_mean) / fitted.x_scale
    cross_values = xz[:, cross]
    own_design = np.column_stack([np.ones(n_rows), xz[:, target]])
    partial_target = target_values - cross_values @ beta_cross_z
    own_coef, *_ = np.linalg.lstsq(own_design, partial_target, rcond=None)
    beta_z = np.zeros(n_features, dtype=float)
    beta_z[cross] = beta_cross_z
    beta_z[target] = float(own_coef[1])
    beta = beta_z / fitted.x_scale
    intercept = float(own_coef[0]) - float(beta @ fitted.x_mean)
    return beta, intercept


def moving_block_pair_indices(
    n_pairs: int,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Resample ordered transition-pair indices in contiguous blocks."""
    if n_pairs < 1:
        raise ValueError("n_pairs must be positive")
    length = min(max(int(block_length), 1), n_pairs)
    n_blocks = int(np.ceil(n_pairs / length))
    starts = rng.integers(0, n_pairs - length + 1, size=n_blocks)
    indices = np.concatenate([start + np.arange(length) for start in starts])
    return indices[:n_pairs]


def fit_linear_dynamics(
    history: pd.DataFrame,
    coef_mode: str,
    max_parents: int = MAX_PARENTS,
    seed: int = 0,
    *,
    features: Sequence[str] | None = None,
) -> LinearDynamics:
    feature_names = tuple(FEATURES if features is None else features)
    values = history[list(feature_names)].to_numpy(dtype=float)
    y = values[1:]
    x = values[:-1]
    k = values.shape[1]
    coef = np.zeros((k, k))
    intercept = np.zeros(k)
    for target in range(k):
        target_values = y[:, target]
        if coef_mode == "dense":
            mean = x.mean(axis=0)
            scale = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
            xz = (x - mean) / scale
            model = LinearRegression().fit(xz, target_values)
            beta_z = np.asarray(model.coef_, dtype=float)
            model_intercept = float(model.intercept_)
            coef[target] = beta_z / scale
            intercept[target] = model_intercept - float(coef[target] @ mean)
        elif coef_mode == "sparse":
            coef[target], intercept[target] = _fit_sparse_target(
                x,
                target_values,
                target,
                max_parents,
                seed,
            )
        else:
            raise ValueError(f"unknown coef_mode {coef_mode!r}")

    coef, intercept, raw_radius, stability_scale = stabilize_lag_matrix(
        coef,
        x,
        y,
    )
    residuals = y - (x @ coef.T + intercept)
    cov = np.cov(residuals, rowvar=False)
    cov = cov + np.eye(k) * (0.05 * np.maximum(np.diag(cov), 1e-6))
    return LinearDynamics(
        coef,
        intercept,
        residuals,
        cov,
        raw_spectral_radius=raw_radius,
        stability_scale=stability_scale,
    )


class LinearPathGenerator:
    """Lag-1 linear rollout with pluggable coefficients and shocks."""

    def __init__(
        self,
        coef_mode: str = "dense",
        shock_mode: str = "gaussian",
        shock_scale: float = 1.0,
        block_length: int = 4,
        max_parents: int = MAX_PARENTS,
        *,
        features: Sequence[str] | None = None,
        bounds: Mapping[str, tuple[float, float]] | None = None,
    ) -> None:
        if shock_mode not in ("gaussian", "bootstrap", "block"):
            raise ValueError(f"unknown shock_mode {shock_mode!r}")
        self.coef_mode = coef_mode
        self.shock_mode = shock_mode
        self.shock_scale = float(shock_scale)
        self.block_length = int(block_length)
        self.max_parents = int(max_parents)
        self.features = tuple(FEATURES if features is None else features)
        self.bounds = SANITY_BOUNDS if bounds is None else dict(bounds)
        if self.block_length < 1:
            raise ValueError("block_length must be positive")
        self.name = f"{coef_mode}_{shock_mode}"
        self.dynamics_: LinearDynamics | None = None
        self._likelihood_cache: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

    def fit(self, history: pd.DataFrame) -> LinearPathGenerator:
        self.dynamics_ = fit_linear_dynamics(
            history,
            self.coef_mode,
            max_parents=self.max_parents,
            features=self.features,
        )
        self._likelihood_cache = {}
        return self

    def _draw_shocks(
        self,
        n_paths: int,
        horizon: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        dynamics = self.dynamics_
        k = dynamics.coef.shape[0]
        if self.shock_mode == "gaussian":
            shocks = rng.multivariate_normal(
                np.zeros(k), dynamics.residual_cov, size=(n_paths, horizon)
            )
        elif self.shock_mode == "bootstrap":
            rows = rng.integers(0, len(dynamics.residuals), size=(n_paths, horizon))
            shocks = dynamics.residuals[rows]
        else:
            # "block": contiguous multi-quarter blocks of historical
            # residual rows, preserving cross-sectional *and* temporal
            # shock dependence (volatility clustering, crisis persistence).
            length = min(self.block_length, len(dynamics.residuals))
            n_blocks = int(np.ceil(horizon / length))
            starts = rng.integers(
                0,
                len(dynamics.residuals) - length + 1,
                size=(n_paths, n_blocks),
            )
            offsets = np.arange(length)
            indices = (starts[:, :, None] + offsets[None, None, :]).reshape(
                n_paths, n_blocks * length
            )[:, :horizon]
            shocks = dynamics.residuals[indices]
        return shocks * self.shock_scale

    def sample(
        self,
        n_paths: int,
        horizon: int,
        rng: np.random.Generator,
        jumpoff: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
    ) -> np.ndarray:
        """Unanchored: dynamic rollout from ``jumpoff`` (shape (k,)).
        Anchored: deviations propagated by the dynamics, added to ``anchor``
        (shape (horizon, k)) -- the challenger mode."""
        if self.dynamics_ is None:
            raise RuntimeError("generator not fitted")
        dynamics = self.dynamics_
        shocks = self._draw_shocks(n_paths, horizon, rng)
        paths = np.empty((n_paths, horizon, dynamics.coef.shape[0]))
        if anchor is not None:
            deviation = np.zeros((n_paths, dynamics.coef.shape[0]))
            for h in range(horizon):
                deviation = deviation @ dynamics.coef.T + shocks[:, h]
                paths[:, h] = anchor[h][None, :] + deviation
        else:
            if jumpoff is None:
                raise RuntimeError("unanchored sampling needs a jumpoff state")
            state = np.tile(jumpoff[None, :], (n_paths, 1))
            for h in range(horizon):
                state = (
                    state @ dynamics.coef.T + dynamics.intercept[None, :] + shocks[:, h]
                )
                paths[:, h] = state
        return clip_to_bounds(paths, self.features, self.bounds)

    def _block_likelihood_parameters(
        self,
        length: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Regularised Gaussian reference law for residual blocks."""
        if length in self._likelihood_cache:
            return self._likelihood_cache[length]
        residuals = self.dynamics_.residuals
        if length > len(residuals):
            raise ValueError("likelihood block exceeds the residual history")
        blocks = np.stack(
            [
                residuals[start : start + length].reshape(-1)
                for start in range(len(residuals) - length + 1)
            ]
        )
        mean = blocks.mean(axis=0)
        covariance = np.atleast_2d(np.cov(blocks, rowvar=False))
        diagonal = np.maximum(np.diag(covariance), 1e-6)
        covariance = covariance + np.diag(0.05 * diagonal + 1e-8)
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            covariance = covariance + np.eye(covariance.shape[0]) * 1e-6
            sign, logdet = np.linalg.slogdet(covariance)
        precision = np.linalg.pinv(covariance)
        parameters = (mean, precision, float(logdet))
        self._likelihood_cache[length] = parameters
        return parameters

    def innovation_loglik(self, path: np.ndarray, jumpoff: np.ndarray) -> float:
        """Block-Gaussian reference likelihood of implied innovations.

        The score uses the fitted mean dynamics and a regularised Gaussian
        approximation to historical residual blocks. For block generators it
        therefore recognises both same-quarter co-movement and within-block
        temporal dependence. It remains a common reference score rather than
        the exact density of the empirical bootstrap distribution.
        """
        if self.dynamics_ is None:
            raise RuntimeError("generator not fitted")
        dynamics = self.dynamics_
        previous = jumpoff
        k = dynamics.coef.shape[0]
        innovations = np.empty((path.shape[0], k), dtype=float)
        for h in range(path.shape[0]):
            predicted = dynamics.coef @ previous + dynamics.intercept
            innovations[h] = path[h] - predicted
            previous = path[h]

        reference_length = self.block_length if self.shock_mode == "block" else 1
        total = 0.0
        for start in range(0, len(innovations), reference_length):
            block = innovations[start : start + reference_length]
            length = len(block)
            mean, precision, logdet = self._block_likelihood_parameters(length)
            innovation = block.reshape(-1) - mean
            dimension = len(innovation)
            total += -0.5 * (
                innovation @ precision @ innovation
                + logdet
                + dimension * np.log(2 * np.pi)
            )
        return float(total)


class StateWeightedBlockGenerator(LinearPathGenerator):
    """Sparse graph with state-weighted historical shock blocks.

    The baseline graph-block generator draws every historical residual block
    uniformly. This challenger keeps the same sparse lagged dynamics, but
    tilts the block draw when the current state is already stressed. Blocks
    whose average residual has an adverse stress signature receive higher
    probability, measured with the fitted residual precision matrix. Calm
    states remain close to uniform, so the change targets crisis persistence
    rather than normal-period forecasting.
    """

    STRESS_SIGNS = {
        "gdp_growth": -1.0,
        "hpi_qoq_growth": -1.0,
        "unemployment": 1.0,
        "bbb_spread": 1.0,
        "vix": 1.0,
        "mortgage_30y": 0.5,
        "fed_funds": 0.0,
    }

    def __init__(
        self,
        block_length: int = 4,
        max_parents: int = MAX_PARENTS,
        temperature: float = 0.8,
        target_scale: float = 0.75,
    ) -> None:
        super().__init__(
            "sparse",
            "block",
            block_length=block_length,
            max_parents=max_parents,
        )
        self.name = "graph_block_weighted"
        self.temperature = float(temperature)
        self.target_scale = float(target_scale)
        self.state_mean_: np.ndarray | None = None
        self.state_scale_: np.ndarray | None = None
        self.blocks_: np.ndarray | None = None
        self.block_means_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None

    def fit(self, history: pd.DataFrame) -> StateWeightedBlockGenerator:
        super().fit(history)
        values = history[FEATURES].to_numpy(dtype=float)
        self.state_mean_ = values.mean(axis=0)
        self.state_scale_ = np.where(values.std(axis=0) > 1e-8, values.std(axis=0), 1.0)
        residuals = self.dynamics_.residuals
        length = min(self.block_length, len(residuals))
        self.blocks_ = np.stack(
            [
                residuals[start : start + length]
                for start in range(len(residuals) - length + 1)
            ]
        )
        self.block_means_ = self.blocks_.mean(axis=1)
        self.precision_ = np.linalg.pinv(self.dynamics_.residual_cov)
        return self

    @property
    def _stress_direction(self) -> np.ndarray:
        return np.asarray(
            [self.STRESS_SIGNS.get(feature, 0.0) for feature in FEATURES],
            dtype=float,
        )

    def _state_severity(self, state: np.ndarray) -> float:
        z = (state - self.state_mean_) / self.state_scale_
        direction = self._stress_direction
        active = np.abs(direction) > 1e-12
        adverse_z = z[active] * np.sign(direction[active])
        severity = float(np.mean(np.maximum(adverse_z, 0.0)))
        return float(np.clip(severity, 0.0, 3.0))

    def _block_probabilities(self, state: np.ndarray) -> np.ndarray:
        severity = self._state_severity(state)
        if severity <= 1e-8 or self.temperature <= 1e-8:
            return np.full(len(self.blocks_), 1.0 / len(self.blocks_))
        residual_std = np.sqrt(np.maximum(np.diag(self.dynamics_.residual_cov), 1e-12))
        target = self.target_scale * severity * self._stress_direction * residual_std
        diff = self.block_means_ - target[None, :]
        distance = np.einsum("ni,ij,nj->n", diff, self.precision_, diff)
        logits = -0.5 * self.temperature * distance
        logits -= float(np.max(logits))
        weights = np.exp(logits)
        return weights / float(weights.sum())

    def _draw_block_for_state(
        self,
        state: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        probabilities = self._block_probabilities(state)
        index = int(rng.choice(len(self.blocks_), p=probabilities))
        return self.blocks_[index]

    def sample(
        self,
        n_paths: int,
        horizon: int,
        rng: np.random.Generator,
        jumpoff: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.dynamics_ is None:
            raise RuntimeError("generator not fitted")
        dynamics = self.dynamics_
        k = dynamics.coef.shape[0]
        paths = np.empty((n_paths, horizon, k))
        if anchor is not None:
            deviation = np.zeros((n_paths, k))
            for path_index in range(n_paths):
                h = 0
                current = anchor[0].copy()
                while h < horizon:
                    block = self._draw_block_for_state(current, rng)
                    for shock in block:
                        if h >= horizon:
                            break
                        deviation[path_index] = (
                            deviation[path_index] @ dynamics.coef.T + shock
                        )
                        paths[path_index, h] = anchor[h] + deviation[path_index]
                        current = paths[path_index, h]
                        h += 1
        else:
            if jumpoff is None:
                raise RuntimeError("unanchored sampling needs a jumpoff state")
            for path_index in range(n_paths):
                h = 0
                state = jumpoff.copy()
                while h < horizon:
                    block = self._draw_block_for_state(state, rng)
                    for shock in block:
                        if h >= horizon:
                            break
                        state = state @ dynamics.coef.T + dynamics.intercept + shock
                        paths[path_index, h] = state
                        h += 1
        return _clip_to_bounds(paths, FEATURES)


class MinnesotaVARGenerator(LinearPathGenerator):
    """Deterministic Minnesota-style generalized-ridge VAR benchmark.

    The serious econometric benchmark for the sparse graph: instead of
    selecting parents, every coefficient is estimated under a prior that
    shrinks own lags toward persistence (prior mean ``delta``) and cross
    lags toward zero, with cross coefficients shrunk harder (``theta``).
    The shrinkage estimate is closed form on standardized data; this is not
    a full posterior-predictive BVAR because coefficient uncertainty is not
    sampled. Hyperparameters are fixed textbook-style
    defaults, fixed before evaluation: delta 0.9, overall tightness
    lambda 0.2, cross-shrinkage theta 0.5.
    """

    def __init__(
        self,
        shock_mode: str = "gaussian",
        block_length: int = 4,
        delta: float = 0.9,
        tightness: float = 0.2,
        theta: float = 0.5,
    ) -> None:
        super().__init__("dense", shock_mode, block_length=block_length)
        self.delta = float(delta)
        self.tightness = float(tightness)
        self.theta = float(theta)
        self.name = f"bvar_{shock_mode}"

    def fit(self, history: pd.DataFrame) -> MinnesotaVARGenerator:
        values = history[FEATURES].to_numpy(dtype=float)
        y_all = values[1:]
        x = values[:-1]
        k = values.shape[1]
        mean = x.mean(axis=0)
        scale = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
        xz = (x - mean) / scale

        coef = np.zeros((k, k))
        intercept = np.zeros(k)
        for target in range(k):
            y = y_all[:, target]
            y_mean, y_scale = y.mean(), max(y.std(), 1e-8)
            yz = (y - y_mean) / y_scale
            # Minnesota prior on standardised coefficients: own lag mean
            # delta, cross means zero; prior std lambda for own lag,
            # lambda*theta for cross lags.
            prior_mean = np.zeros(k)
            prior_mean[target] = self.delta
            prior_sd = np.full(k, self.tightness * self.theta)
            prior_sd[target] = self.tightness
            noise_var = max(1.0 - self.delta**2, 1e-4)  # rough residual scale
            penalty = noise_var / prior_sd**2
            gram = xz.T @ xz + np.diag(penalty)
            beta_z = np.linalg.solve(gram, xz.T @ yz + penalty * prior_mean)
            coef[target] = beta_z * (y_scale / scale)
            intercept[target] = y_mean - float(coef[target] @ mean)

        coef, intercept, raw_radius, stability_scale = stabilize_lag_matrix(
            coef,
            x,
            y_all,
        )
        residuals = y_all - (x @ coef.T + intercept)
        cov = np.cov(residuals, rowvar=False)
        cov = cov + np.eye(k) * (0.05 * np.maximum(np.diag(cov), 1e-6))
        self.dynamics_ = LinearDynamics(
            coef,
            intercept,
            residuals,
            cov,
            raw_spectral_radius=raw_radius,
            stability_scale=stability_scale,
        )
        return self


class SparseMinnesotaVARGenerator(MinnesotaVARGenerator):
    """Sparse graph with Minnesota shrinkage on the retained lag edges.

    The Lasso graph determines which cross-variable lag coefficients may be
    nonzero. The retained coefficients are then re-estimated as a Minnesota
    posterior mean: own lags shrink toward persistence and cross lags shrink
    toward zero. This isolates coefficient shrinkage from graph sparsity and
    the shock mechanism.
    """

    def __init__(
        self,
        shock_mode: str = "block",
        block_length: int = 4,
        max_parents: int = MAX_PARENTS,
        delta: float = 0.9,
        tightness: float = 0.2,
        theta: float = 0.5,
    ) -> None:
        super().__init__(
            shock_mode=shock_mode,
            block_length=block_length,
            delta=delta,
            tightness=tightness,
            theta=theta,
        )
        self.max_parents = int(max_parents)
        self.name = f"graph_bvar_{shock_mode}"

    def fit(self, history: pd.DataFrame) -> SparseMinnesotaVARGenerator:
        values = history[FEATURES].to_numpy(dtype=float)
        y_all = values[1:]
        x = values[:-1]
        k = values.shape[1]
        mean = x.mean(axis=0)
        scale = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
        xz = (x - mean) / scale

        # Graph selection is identical to graph_block. Own persistence is
        # always retained even when the Lasso sets that coefficient to zero.
        selected_dynamics = fit_linear_dynamics(
            history,
            "sparse",
            max_parents=self.max_parents,
        )
        support = np.abs(selected_dynamics.coef) > 1e-12
        np.fill_diagonal(support, True)

        coef = np.zeros((k, k))
        intercept = np.zeros(k)
        for target in range(k):
            y = y_all[:, target]
            y_mean, y_scale = y.mean(), max(y.std(), 1e-8)
            yz = (y - y_mean) / y_scale
            selected = np.flatnonzero(support[target])
            own_position = int(np.flatnonzero(selected == target)[0])

            prior_mean = np.zeros(len(selected))
            prior_mean[own_position] = self.delta
            prior_sd = np.full(len(selected), self.tightness * self.theta)
            prior_sd[own_position] = self.tightness
            noise_var = max(1.0 - self.delta**2, 1e-4)
            penalty = noise_var / prior_sd**2

            x_selected = xz[:, selected]
            gram = x_selected.T @ x_selected + np.diag(penalty)
            beta_selected = np.linalg.solve(
                gram,
                x_selected.T @ yz + penalty * prior_mean,
            )
            beta_z = np.zeros(k)
            beta_z[selected] = beta_selected
            coef[target] = beta_z * (y_scale / scale)
            intercept[target] = y_mean - float(coef[target] @ mean)

        coef, intercept, raw_radius, stability_scale = stabilize_lag_matrix(
            coef,
            x,
            y_all,
        )
        residuals = y_all - (x @ coef.T + intercept)
        cov = np.cov(residuals, rowvar=False)
        cov = cov + np.eye(k) * (0.05 * np.maximum(np.diag(cov), 1e-6))
        self.dynamics_ = LinearDynamics(
            coef,
            intercept,
            residuals,
            cov,
            raw_spectral_radius=raw_radius,
            stability_scale=stability_scale,
        )
        return self


class RandomBoundedGenerator:
    """The purest strawman: random but bounded.

    Each variable, each quarter, drawn independently and uniformly within
    the range observed up to the jump-off (information discipline is kept:
    bounds come from then-available history only). No persistence, no
    co-movement, no dynamics. Exists to demonstrate that bounds alone do
    not make scenarios - every value is individually "allowed" while the
    joint path is calibrated to nothing.
    """

    name = "random_bounded"

    def __init__(self) -> None:
        self.lo_: np.ndarray | None = None
        self.hi_: np.ndarray | None = None

    def fit(self, history: pd.DataFrame) -> RandomBoundedGenerator:
        values = history[FEATURES].to_numpy(dtype=float)
        self.lo_ = values.min(axis=0)
        self.hi_ = values.max(axis=0)
        return self

    def sample(
        self,
        n_paths: int,
        horizon: int,
        rng: np.random.Generator,
        jumpoff: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.lo_ is None:
            raise RuntimeError("generator not fitted")
        paths = rng.uniform(self.lo_, self.hi_, size=(n_paths, horizon, len(FEATURES)))
        return _clip_to_bounds(paths, FEATURES)


class DeltaBootstrapGenerator:
    """Structureless floor: random walk with bootstrapped historical changes."""

    name = "delta_bootstrap"

    def __init__(self) -> None:
        self.deltas_: np.ndarray | None = None

    def fit(self, history: pd.DataFrame) -> DeltaBootstrapGenerator:
        values = history[FEATURES].to_numpy(dtype=float)
        self.deltas_ = np.diff(values, axis=0)
        return self

    def sample(
        self,
        n_paths: int,
        horizon: int,
        rng: np.random.Generator,
        jumpoff: np.ndarray | None = None,
        anchor: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.deltas_ is None:
            raise RuntimeError("generator not fitted")
        rows = rng.integers(0, len(self.deltas_), size=(n_paths, horizon))
        steps = self.deltas_[rows]
        if anchor is not None:
            cumulative = np.cumsum(
                steps - steps.mean(axis=(0, 1), keepdims=True), axis=1
            )
            paths = anchor[None, :, :] + cumulative
        else:
            if jumpoff is None:
                raise RuntimeError("unanchored sampling needs a jumpoff state")
            paths = jumpoff[None, None, :] + np.cumsum(steps, axis=1)
        return _clip_to_bounds(paths, FEATURES)


def default_generator_suite(block_length: int = 4) -> dict[str, object]:
    """The evaluation suite: floor, incumbent, thesis generator, ablations.

    The shock ladder gaussian -> bootstrap -> block isolates distribution
    shape, cross-sectional co-movement, and temporal persistence as three
    separately measured ingredients.
    """
    return {
        "delta_bootstrap": DeltaBootstrapGenerator(),
        "var_gaussian": LinearPathGenerator("dense", "gaussian"),
        "var_bootstrap": LinearPathGenerator("dense", "bootstrap"),
        "var_block": LinearPathGenerator("dense", "block", block_length=block_length),
        "graph_gaussian": LinearPathGenerator("sparse", "gaussian"),
        "graph_bootstrap": LinearPathGenerator("sparse", "bootstrap"),
        "graph_block": LinearPathGenerator(
            "sparse", "block", block_length=block_length
        ),
    }


def build_generator(name: str, block_length: int = 4):
    """Construct one maintained legacy linear-generator diagnostic."""
    suite = default_generator_suite(block_length=block_length)
    suite["random_bounded"] = RandomBoundedGenerator()
    if name not in suite:
        raise ValueError(
            f"unknown generator {name!r}; available generators: {sorted(suite)}"
        )
    return suite[name]


def edge_confidence(
    history: pd.DataFrame,
    coef_mode: str = "sparse",
    n_boot: int = 100,
    seed: int = 7,
    block_length: int = 4,
) -> pd.DataFrame:
    """Bootstrap edge-confidence for the learned graph (after Gao et al.).

    Refits the dynamics on moving-block resamples of the ordered (lagged
    state, next state) pairs and reports, per directed edge, the share of
    refits in which it survives with a non-zero coefficient. These are
    dependence-aware stability frequencies, not causal confidence levels.
    """
    rng = np.random.default_rng(seed)
    values = history[FEATURES].to_numpy(dtype=float)
    n_pairs = len(values) - 1
    counts = np.zeros((len(FEATURES), len(FEATURES)))
    coef_sum = np.zeros_like(counts)
    for _ in range(n_boot):
        chosen = moving_block_pair_indices(n_pairs, block_length, rng)
        x = values[chosen]
        y = values[chosen + 1]
        dynamics = _fit_from_pairs(x, y, coef_mode, seed=int(rng.integers(1e6)))
        nonzero = np.abs(dynamics.coef) > 1e-10
        counts += nonzero
        coef_sum += dynamics.coef
    rows = []
    for target_index, target in enumerate(FEATURES):
        for parent_index, parent in enumerate(FEATURES):
            if counts[target_index, parent_index] == 0:
                continue
            rows.append(
                {
                    "parent": parent,
                    "target": target,
                    "confidence": counts[target_index, parent_index] / n_boot,
                    "mean_coef": coef_sum[target_index, parent_index] / n_boot,
                }
            )
    return pd.DataFrame(rows).sort_values(
        "confidence", ascending=False, ignore_index=True
    )


def _fit_from_pairs(
    x: np.ndarray,
    y: np.ndarray,
    coef_mode: str,
    max_parents: int = MAX_PARENTS,
    seed: int = 0,
) -> LinearDynamics:
    k = x.shape[1]
    mean = x.mean(axis=0)
    scale = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
    xz = (x - mean) / scale
    coef = np.zeros((k, k))
    intercept = np.zeros(k)
    for target in range(k):
        if coef_mode == "dense":
            model = LinearRegression().fit(xz, y[:, target])
            beta_z = np.asarray(model.coef_, dtype=float)
            model_intercept = float(model.intercept_)
        else:
            beta_z, model_intercept = _fit_sparse_target(
                xz,
                y[:, target],
                target,
                max_parents,
                seed,
            )
        coef[target] = beta_z / scale
        intercept[target] = model_intercept - float(coef[target] @ mean)
    coef, intercept, raw_radius, stability_scale = stabilize_lag_matrix(
        coef,
        x,
        y,
    )
    residuals = y - (x @ coef.T + intercept)
    cov = np.cov(residuals, rowvar=False)
    cov = cov + np.eye(k) * (0.05 * np.maximum(np.diag(cov), 1e-6))
    return LinearDynamics(
        coef,
        intercept,
        residuals,
        cov,
        raw_spectral_radius=raw_radius,
        stability_scale=stability_scale,
    )


__all__ = [
    "DeltaBootstrapGenerator",
    "LinearDynamics",
    "LinearPathGenerator",
    "MinnesotaVARGenerator",
    "RandomBoundedGenerator",
    "SparseMinnesotaVARGenerator",
    "StateWeightedBlockGenerator",
    "build_generator",
    "clip_to_bounds",
    "default_generator_suite",
    "edge_confidence",
    "fit_linear_dynamics",
    "moving_block_pair_indices",
    "stabilize_lag_matrix",
]
