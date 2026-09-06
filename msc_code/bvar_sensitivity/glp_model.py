"""Conjugate BVAR with fixed or GLP-inspired MAP shrinkage."""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln, multigammaln
from scipy.stats import invwishart

from src.bvar import System, _spd


def _posterior(X, Y, strength):
    """Conjugate posterior for B | Sigma ~ MN(B0, V0, Sigma)."""
    n, k = Y.shape
    B0 = np.zeros((k + 1, k))
    B0[1:] = 0.9 * np.eye(k)
    variances = np.r_[100.0, np.full(k, strength**2)]
    V0, P0 = np.diag(variances), np.diag(1.0 / variances)
    S0, nu0 = np.eye(k), float(k + 2)
    precision = P0 + X.T @ X
    Vn = np.linalg.solve(precision, np.eye(k + 1))
    Vn = 0.5 * (Vn + Vn.T)
    Bn = np.linalg.solve(precision, P0 @ B0 + X.T @ Y)
    residual, shift = Y - X @ Bn, Bn - B0
    Sn = S0 + residual.T @ residual + shift.T @ P0 @ shift
    Sn = 0.5 * (Sn + Sn.T)
    return dict(Bn=Bn, Vn=Vn, Sn=Sn, V0=V0, S0=S0,
                nu0=nu0, nun=nu0 + n, n=n, k=k)


def _logdet(matrix):
    sign, value = np.linalg.slogdet(matrix)
    if sign <= 0 or not np.isfinite(value):
        raise np.linalg.LinAlgError("expected positive definite matrix")
    return float(value)


def _log_evidence(X, Y, strength):
    p = _posterior(X, Y, strength)
    n, k, nu0, nun = p["n"], p["k"], p["nu0"], p["nun"]
    return float(-n * k / 2.0 * np.log(np.pi)
                 + k / 2.0 * (_logdet(p["Vn"]) - _logdet(p["V0"]))
                 + nu0 / 2.0 * _logdet(p["S0"])
                 - nun / 2.0 * _logdet(p["Sn"])
                 + multigammaln(nun / 2.0, k) - multigammaln(nu0 / 2.0, k))


def _log_lambda_prior(strength):
    # Gamma hyperprior with mode 0.2 and standard deviation 0.4.
    mode, sd = 0.2, 0.4
    scale = 2.0 * sd**2 / (np.sqrt(mode**2 + 4.0 * sd**2) + mode)
    shape = 1.0 + mode / scale
    return float((shape - 1.0) * np.log(strength) - strength / scale
                 - shape * np.log(scale) - gammaln(shape))


def _select_lambda(X, Y):
    # Optimise density in lambda using log coordinates, without a Jacobian.
    def objective(log_lambda):
        strength = float(np.exp(log_lambda))
        return _log_evidence(X, Y, strength) + _log_lambda_prior(strength)

    grid = np.unique(np.r_[np.linspace(np.log(0.001), np.log(3.0), 41), np.log(0.2)])
    scores = np.asarray([objective(value) for value in grid])
    if not np.isfinite(scores).all():
        raise FloatingPointError("nonfinite shrinkage objective")
    best = int(np.argmax(scores))
    result = minimize_scalar(lambda z: -objective(z), method="bounded",
                             bounds=(grid[max(0, best - 1)],
                                     grid[min(len(grid) - 1, best + 1)]),
                             options={"xatol": 1e-7, "maxiter": 200})
    selected, score = float(grid[best]), float(scores[best])
    if result.success and np.isfinite(result.fun) and -result.fun > score:
        selected = float(result.x)
    return float(np.exp(selected))


def fit_conjugate_bvar(values, seed, optimize=False, n_draws=100,
                       max_radius=0.995, max_iter=200000):
    """Return posterior systems and shrinkage/sampling diagnostics."""
    values = np.asarray(values, float)
    if (values.ndim != 2 or len(values) < 24 or values.shape[1] < 1
            or not np.isfinite(values).all()):
        raise ValueError("values must contain at least 24 finite rows and one variable")
    if (not isinstance(n_draws, (int, np.integer)) or n_draws < 1
            or not isinstance(max_iter, (int, np.integer)) or max_iter < n_draws):
        raise ValueError("draw counts must be positive integers with max_iter >= n_draws")
    if not np.isfinite(max_radius) or not 0 < max_radius < 1:
        raise ValueError("max_radius must lie between zero and one")

    # Standardise predictors and targets separately using the training sample.
    x, y = values[:-1], values[1:]
    x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
    x_sd = np.where(x.std(axis=0) > 1e-8, x.std(axis=0), 1.0)
    y_sd = np.where(y.std(axis=0) > 1e-8, y.std(axis=0), 1.0)
    X = np.column_stack([np.ones(len(x)), (x - x_mean) / x_sd])
    Y = (y - y_mean) / y_sd
    strength = _select_lambda(X, Y) if optimize else 0.2
    p = _posterior(X, Y, strength)
    rng = np.random.default_rng(seed)
    row_root = np.linalg.cholesky(p["Vn"])
    draws, attempted = [], 0

    # Draw Sigma, then B conditional on Sigma, from the conjugate posterior.
    while len(draws) < n_draws and attempted < max_iter:
        sigma = np.atleast_2d(invwishart.rvs(df=p["nun"], scale=p["Sn"], random_state=rng))
        column_root = np.linalg.cholesky(sigma)
        B = p["Bn"] + row_root @ rng.standard_normal(p["Bn"].shape) @ column_root.T
        B = np.asfortranarray(B)
        A = B[1:].T * y_sd[:, None] / x_sd[None, :]
        c = y_mean + y_sd * B[0] - A @ x_mean
        residuals = y - (x @ A.T + c[None, :])
        rescale = np.diag(y_sd)
        sigma_raw = _spd(rescale @ sigma @ rescale)
        attempted += 1
        # Retain only stable systems on the original measurement scales.
        if np.max(np.abs(np.linalg.eigvals(A))) < max_radius:
            draws.append(System(c, A, sigma_raw, residuals))
    if len(draws) != n_draws:
        raise RuntimeError(f"only {len(draws)} stable posterior draws after {attempted} attempts")
    return draws, {"lambda": float(strength), "attempted_draws": attempted}
