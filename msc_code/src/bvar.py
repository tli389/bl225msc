"""Fixed-prior Minnesota BVAR -- the benchmark as the report describes it, in ~90 lines.

``check_benchmarks.py`` reproduces every BVAR row of the report from it.

Report step -> code
  1. arrange the training history into one-quarter transition pairs and
     standardise lagged predictors and next-quarter targets separately
  2. the Minnesota prior: own lag ~ N(delta, 0.2^2), cross lags ~ N(0, 0.1^2),
     intercept ~ N(0, 10^2), Sigma ~ IW(K+2, I_K)
  3. semi-conjugate Gibbs: draw the coefficients given Sigma (a Gaussian
     regression draw whose precision is data precision + prior precision),
     then Sigma given the coefficients (inverse-Wishart from the residuals)
  4. discard 150 burn-in iterations, keep every third draw, reject draws whose
     lag matrix has spectral radius >= 0.995, stop at 100 retained systems
  5. map each retained system back to the original measurement scales
Generation (unconditional): pick one retained system uniformly, hold it for
the whole horizon, add Gaussian shocks N(0, Sigma) independently each quarter.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import invwishart


class System:
    """One retained posterior system (c, A, Sigma) on the original scales."""

    def __init__(self, intercept, coef, residual_cov, residuals):
        self.intercept, self.coef = intercept, coef
        self.residual_cov, self.residuals = residual_cov, residuals


def _spd(m, floor=1e-10):
    """Symmetrise and floor the eigenvalues (numerical hygiene only)."""
    m = 0.5 * (m + m.T)
    vals, vecs = np.linalg.eigh(m)
    return (vecs * np.maximum(vals, floor)[None, :]) @ vecs.T


def fit_minnesota_bvar(values, seed, own_lag_means=None, delta=0.9, own_sd=0.2,
                       cross_sd=0.1, intercept_sd=10.0, n_draws=100,
                       burn_in=150, thin=3, max_radius=0.995, max_iter=20000):
    """values: (T, K) training history in feature order.  Returns 100 Systems."""
    values = np.asarray(values, float)
    T_all, K = values.shape

    # 1. transition pairs, standardised separately for predictors and targets
    x, y = values[:-1], values[1:]
    x_mean, y_mean = x.mean(0), y.mean(0)
    x_sd = np.where(x.std(0) > 1e-8, x.std(0), 1.0)
    y_sd = np.where(y.std(0) > 1e-8, y.std(0), 1.0)
    X = np.column_stack([np.ones(len(x)), (x - x_mean) / x_sd])   # (T, K+1)
    Y = (y - y_mean) / y_sd                                        # (T, K)
    T = len(Y)

    # 2. the Minnesota prior: one anchor and one sd per coefficient
    #    rows = intercept, then the K lagged predictors; columns = equations
    prior_mean = np.zeros((K + 1, K))
    prior_sd = np.full((K + 1, K), cross_sd)
    prior_sd[0] = intercept_sd
    own = np.full(K, delta) if own_lag_means is None else np.asarray(own_lag_means, float)
    for k in range(K):
        prior_mean[k + 1, k] = own[k]
        prior_sd[k + 1, k] = own_sd
    m0 = prior_mean.reshape(-1, order="F")
    prec0 = 1.0 / prior_sd.reshape(-1, order="F") ** 2
    prior_df = K + 2
    prior_scale = np.eye(K) * (prior_df - K - 1)   # = I_K: prior mean of Sigma is I

    # 3. Gibbs sampling, alternating the two conditional draws
    rng = np.random.default_rng(seed + T_all)
    XtX = X.T @ X
    sigma = np.eye(K)
    draws, it = [], 0
    while len(draws) < n_draws:
        it += 1
        if it > max_iter:
            raise RuntimeError("too few stable posterior draws")
        # coefficients | Sigma: precision = kron(Sigma^-1, X'X) + prior precision
        s_inv = np.linalg.inv(_spd(sigma))
        prec = np.kron(s_inv, XtX) + np.diag(prec0)
        rhs = (X.T @ Y @ s_inv).reshape(-1, order="F") + prec0 * m0
        prec = 0.5 * (prec + prec.T)
        root = np.linalg.cholesky(prec)
        mean = np.linalg.solve(root.T, np.linalg.solve(root, rhs))
        beta = mean + np.linalg.solve(root.T, rng.standard_normal(len(mean)))
        # Sigma | coefficients: inverse-Wishart from the residual scatter
        B = beta.reshape(K + 1, K, order="F")
        R = Y - X @ B
        sigma = np.atleast_2d(invwishart.rvs(df=prior_df + T,
                                             scale=_spd(prior_scale + R.T @ R),
                                             random_state=rng))
        # 4. burn-in, thinning, stability
        if it <= burn_in or (it - burn_in - 1) % thin:
            continue
        # 5. back to the original scales
        A = B[1:].T * y_sd[:, None] / x_sd[None, :]
        c = y_mean + y_sd * B[0] - A @ x_mean
        if np.max(np.abs(np.linalg.eigvals(A))) >= max_radius:
            continue
        sigma_raw = _spd(np.diag(y_sd) @ sigma @ np.diag(y_sd))
        residuals = y - (x @ A.T + c)
        draws.append(System(c, A, sigma_raw, residuals))
    return draws


def sample_unconditional(draws, n_paths, horizon, rng, jumpoff):
    """Pick a system uniformly per path; Gaussian shocks N(0, Sigma) each quarter."""
    K = len(jumpoff)
    paths = np.empty((n_paths, horizon, K))
    which = rng.integers(0, len(draws), size=n_paths)
    for m in range(n_paths):
        s = draws[which[m]]
        state = np.asarray(jumpoff, float)
        for h in range(horizon):
            state = s.intercept + s.coef @ state + rng.multivariate_normal(
                np.zeros(K), s.residual_cov)
            paths[m, h] = state
    return paths
