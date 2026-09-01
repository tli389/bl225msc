"""Dense Gaussian VAR -- the benchmark as the report describes it.

  1. dense VAR(1) by ordinary least squares, one equation per variable
  2. spectral radius above 0.995 -> lag matrix rescaled uniformly and the
     intercept re-centred so the average one-quarter transition is preserved
  3. residual covariance regularised by adding 5% of each residual variance
     to its diagonal
Shocks are Gaussian N(0, Sigma), independent across quarters; conditional
completion conditions the twelve-quarter Gaussian path analytically (the
repository's Gaussian-law sampler does both, so this file only fits).
"""

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

    # 1. OLS on standardised lagged predictors, mapped back to raw units
    mean = x.mean(0)
    sd = np.where(x.std(0) > 1e-8, x.std(0), 1.0)
    design = np.column_stack([np.ones(len(x)), (x - mean) / sd])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)      # (K+1, K)
    coef = beta[1:].T / sd[None, :]                          # row k = equation k
    intercept = beta[0] - coef @ mean

    # 2. stability: rescale the lag matrix, re-centre the intercept
    radius = float(np.max(np.abs(np.linalg.eigvals(coef))))
    if radius > max_radius:
        coef = coef * (max_radius / radius)
    intercept = y.mean(0) - x.mean(0) @ coef.T

    # 3. residuals and the regularised one-quarter covariance
    residuals = y - (x @ coef.T + intercept)
    cov = np.cov(residuals, rowvar=False)
    cov = cov + np.eye(K) * (0.05 * np.maximum(np.diag(cov), 1e-6))
    return System(intercept, coef, cov, residuals)
