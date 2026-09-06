"""Dense Gaussian VAR fitting with a stability cap."""

from __future__ import annotations

import numpy as np


class System:
    def __init__(self, intercept, coef, residual_cov, residuals):
        self.intercept, self.coef = intercept, coef
        self.residual_cov, self.residuals = residual_cov, residuals


def fit_gaussian_var(values, max_radius=0.995):
    """values: (T, K) training history.  Returns one System (c, A, Sigma)."""
    values = np.asarray(values, float)
    x, y = values[:-1], values[1:]
    K = values.shape[1]

    # Fit all lagged relationships by OLS and return to original units.
    mean = x.mean(0)
    sd = np.where(x.std(0) > 1e-8, x.std(0), 1.0)
    design = np.column_stack([np.ones(len(x)), (x - mean) / sd])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)      # (K+1, K)
    coef = beta[1:].T / sd[None, :]                          # row k = equation k
    intercept = beta[0] - coef @ mean

    # Apply the stability cap and recover the intercept.
    radius = float(np.max(np.abs(np.linalg.eigvals(coef))))
    if radius > max_radius:
        coef = coef * (max_radius / radius)
    intercept = y.mean(0) - x.mean(0) @ coef.T

    # Calculate residuals and regularise their one-quarter covariance.
    residuals = y - (x @ coef.T + intercept)
    cov = np.cov(residuals, rowvar=False)
    cov = cov + np.eye(K) * (0.05 * np.maximum(np.diag(cov), 1e-6))
    return System(intercept, coef, cov, residuals)
