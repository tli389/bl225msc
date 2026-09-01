"""GIB-VAR -- the generator of the report's design chapter.

Written to follow the report's description step by step so the whole backbone
can be read in one sitting; check_benchmarks.py reproduces every GIB-VAR
row of the report from it.

Report step -> code
  Graph-weighted estimation, one equation per variable      (_fit_equation)
    * lagged predictors standardised; intercept and own lag partialled out
      (Frisch-Waugh-Lovell) so they carry no penalty        (_partial_out)
    * cross columns rescaled by 1 / w_kj (w = 0.35 preferred, 1 otherwise) so
      one ordinary Lasso solves the weighted objective; beta = gamma / w
    * lambda chosen by forward validation: 60 log-spaced ratios of lambda_max,
      2-5 chronological folds, preprocessing redone inside each fold, the
      ratio with the lowest validation error kept               (_select_ratio)
    * at most three cross lags: keep the three largest, refit at the same
      lambda; intercept and own lag recovered by least squares
  Spectral cap: rho(A) > 0.995 -> A* = 0.995 A / rho(A), c* = x_bar' - A* x_bar
  Bootstrap bank: 25 refits on non-circular four-quarter blocks of transition
    pairs; residuals recomputed on the chronological sample for each system;
    the pool mean moved into the intercept (c = c_hat + r_bar, e = r - r_bar);
    Sigma = pool covariance + 5% diagonal ridge                  (fit_gibvar)
  Unconditional generation: one system per path, four-quarter residual blocks
    resampled from that system's own pool           (sample_unconditional)
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import Lasso, lars_path
from sklearn.model_selection import TimeSeriesSplit


class System:
    def __init__(self, intercept, coef, residual_cov, residuals):
        self.intercept, self.coef = intercept, coef
        self.residual_cov, self.residuals = residual_cov, residuals


# ----------------------------------------------------------- one equation
def _partial_out(x, y, target, weights):
    """Standardise, project out [1, own lag], rescale cross columns by 1/w."""
    n, K = x.shape
    cross = np.array([j for j in range(K) if j != target])
    mean = x.mean(0)
    scale = np.where(x.std(0) > 1e-8, x.std(0), 1.0)
    z = (x - mean) / scale
    own = np.column_stack([np.ones(n), z[:, target]])
    y_proj, *_ = np.linalg.lstsq(own, y, rcond=None)
    x_proj, *_ = np.linalg.lstsq(own, z[:, cross], rcond=None)
    y_res = y - own @ y_proj
    x_res = z[:, cross] - own @ x_proj
    res_scale = np.where(x_res.std(0) > 1e-8, x_res.std(0), 1.0)
    return dict(cross=cross, mean=mean, scale=scale, y_proj=y_proj, x_proj=x_proj,
                res_scale=res_scale, weights=weights,
                design=x_res / res_scale / weights[None, :], y_res=y_res)


def _apply(fit, x, target):
    """Apply a training fold's transformation to its validation rows."""
    z = (x - fit["mean"]) / fit["scale"]
    own = np.column_stack([np.ones(len(x)), z[:, target]])
    baseline = own @ fit["y_proj"]
    x_res = z[:, fit["cross"]] - own @ fit["x_proj"]
    return x_res / fit["res_scale"] / fit["weights"][None, :], baseline


def _select_ratio(x, y, target, weights):
    """Forward validation of lambda / lambda_max over 60 log-spaced ratios."""
    ratios = np.geomspace(1.0, 1e-3, 60)
    losses = np.zeros(len(ratios))
    n_splits = min(5, max(2, len(x) // 12))
    folds = list(TimeSeriesSplit(n_splits=n_splits).split(x))
    for tr, va in folds:
        fit = _partial_out(x[tr], y[tr], target, weights)
        design_va, baseline = _apply(fit, x[va], target)
        alpha_max = float(np.max(np.abs(fit["design"].T @ fit["y_res"])) / max(len(tr), 1))
        if alpha_max <= 1e-12 or np.std(fit["y_res"]) <= 1e-10:
            pred = np.repeat(baseline[:, None], len(ratios), axis=1)
        else:
            alphas = np.maximum(alpha_max * ratios, 1e-12)
            path_alphas, _, path_coefs = lars_path(fit["design"], fit["y_res"],
                                                   method="lasso", alpha_min=0.0)
            order = np.argsort(path_alphas)
            uniq, idx = np.unique(path_alphas[order], return_index=True)
            ordered = path_coefs[:, order][:, idx]
            coefs = np.vstack([np.interp(alphas, uniq, row) for row in ordered])
            pred = baseline[:, None] + design_va @ coefs
        losses += np.mean((y[va, None] - pred) ** 2, axis=0)
    return float(ratios[int(np.argmin(losses / len(folds)))])


def _fit_equation(x, y, target, weights, max_parents=3):
    """Weighted Lasso with unpenalised intercept and own lag; <= 3 parents."""
    n, K = x.shape
    ratio = _select_ratio(x, y, target, weights)
    fit = _partial_out(x, y, target, weights)
    gamma = np.zeros(K - 1)
    if np.std(fit["y_res"]) > 1e-10:
        alpha_max = float(np.max(np.abs(fit["design"].T @ fit["y_res"])) / max(n, 1))
        lasso = Lasso(alpha=max(alpha_max * ratio, 1e-12), fit_intercept=False,
                      max_iter=100_000).fit(fit["design"], fit["y_res"])
        ranked = np.argsort(np.abs(lasso.coef_))[::-1]
        keep = [int(j) for j in ranked if abs(float(lasso.coef_[j])) > 1e-12][:max_parents]
        if keep:
            refit = Lasso(alpha=float(lasso.alpha), fit_intercept=False,
                          max_iter=100_000).fit(fit["design"][:, keep], fit["y_res"])
            gamma[keep] = refit.coef_
    beta_cross = gamma / fit["weights"] / fit["res_scale"]        # beta = gamma / w
    z = (x - fit["mean"]) / fit["scale"]
    own = np.column_stack([np.ones(n), z[:, target]])
    own_coef, *_ = np.linalg.lstsq(own, y - z[:, fit["cross"]] @ beta_cross, rcond=None)
    beta_z = np.zeros(K)
    beta_z[fit["cross"]] = beta_cross
    beta_z[target] = own_coef[1]
    beta = beta_z / fit["scale"]
    return beta, float(own_coef[0]) - float(beta @ fit["mean"])


# ------------------------------------------------------------ one system
def _regularised_cov(residuals):
    cov = np.atleast_2d(np.cov(residuals, rowvar=False))
    return cov + np.diag(0.05 * np.maximum(np.diag(cov), 1e-6) + 1e-8)


def fit_system(x, y, features, graph, weight=0.35, max_parents=3, max_radius=0.995):
    """Fit the sparse VAR(1) to transition arrays (x -> y); cap; centre."""
    K = x.shape[1]
    coef, intercept = np.zeros((K, K)), np.zeros(K)
    for k in range(K):
        preferred = set(graph.get(features[k], ()))
        weights = np.array([weight if features[j] in preferred else 1.0
                            for j in range(K) if j != k])
        coef[k], intercept[k] = _fit_equation(x, y[:, k], k, weights, max_parents)
    radius = float(np.max(np.abs(np.linalg.eigvals(coef))))
    coef = coef * (1.0 if radius <= max_radius else max_radius / radius)
    intercept = y.mean(0) - x.mean(0) @ coef.T                  # re-centred
    residuals = y - (x @ coef.T + intercept)
    cov = _regularised_cov(residuals)
    shift = residuals.mean(0)                                    # mean into intercept
    return System(intercept + shift, coef, cov, residuals - shift)


def _block_pair_indices(n_pairs, block, rng):
    """Non-circular blocks of consecutive transition pairs, truncated to n."""
    length = min(block, n_pairs)
    starts = rng.integers(0, n_pairs - length + 1, size=int(np.ceil(n_pairs / length)))
    return np.concatenate([s + np.arange(length) for s in starts])[:n_pairs]


def fit_gibvar(values, features, graph, seed, n_systems=25, block=4, **kw):
    """The bank: 25 refits on block-resampled histories, chronological pools."""
    values = np.asarray(values, float)
    x, y = values[:-1], values[1:]
    rng = np.random.default_rng(seed + len(values))
    bank = []
    for _ in range(n_systems):
        idx = _block_pair_indices(len(x), block, rng)
        cand = fit_system(values[idx], values[idx + 1], features, graph, **kw)
        r = y - (x @ cand.coef.T + cand.intercept)                  # chronological
        cov = _regularised_cov(r)
        shift = r.mean(0)
        bank.append(System(cand.intercept + shift, cand.coef, cov, r - shift))
    return bank


# ----------------------------------------------------- unconditional paths
def sample_unconditional(bank, n_paths, horizon, rng, jumpoff, block=4):
    """One system per path; four-quarter residual blocks from its own pool."""
    K = len(jumpoff)
    assign = rng.integers(0, len(bank), size=n_paths)
    paths = np.empty((n_paths, horizon, K))
    for b in np.unique(assign):
        mask = assign == b
        pool = bank[b].residuals
        length = min(block, len(pool), horizon)
        n_blocks = int(np.ceil(horizon / length))
        starts = rng.integers(0, len(pool) - length + 1, size=(int(mask.sum()), n_blocks))
        idx = (starts[:, :, None] + np.arange(length)[None, None, :]).reshape(
            int(mask.sum()), -1)[:, :horizon]
        shocks = pool[idx]
        state = np.tile(np.asarray(jumpoff, float)[None, :], (int(mask.sum()), 1))
        for h in range(horizon):
            state = state @ bank[b].coef.T + bank[b].intercept + shocks[:, h]
            paths[mask, h] = state
    return paths
