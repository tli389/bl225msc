"""Gaussian path moments, conditioning and mixture sampling."""

from __future__ import annotations

import numpy as np


def _eig_floor(cov, rel=1e-9):
    cov = 0.5 * (cov + cov.T)
    vals, vecs = np.linalg.eigh(cov)
    return np.maximum(vals, rel * max(float(vals.max()), 1.0)), vecs


def joint_moments(system, jumpoff, horizon):
    """Mean and covariance of the stacked H*K path given the jump-off."""
    A, c, sigma = system.coef, system.intercept, system.residual_cov
    K = len(c)
    mu, P, state, cov = [], [], np.asarray(jumpoff, float), np.zeros((K, K))
    # No-shock mean and within-quarter forecast covariance.
    for _ in range(horizon):
        state = A @ state + c
        cov = A @ cov @ A.T + sigma
        mu.append(state)
        P.append(cov.copy())
    omega = np.zeros((horizon * K, horizon * K))
    for h in range(horizon):
        block = slice(h * K, (h + 1) * K)
        omega[block, block] = P[h]
        cross = P[h]
        # Propagate covariance across quarters: Omega_(later,h) = A^(later-h) P_h.
        for later in range(h + 1, horizon):
            cross = A @ cross
            lb = slice(later * K, (later + 1) * K)
            omega[lb, block], omega[block, lb] = cross, cross.T
    return np.concatenate(mu), omega


def complete(bank, supplied, obs_cols, n_paths, rng, jumpoff):
    """supplied: (H, K) with the supplied columns filled; obs_cols: their indices."""
    H, K = supplied.shape
    obs = np.array([h * K + j for h in range(H) for j in obs_cols])
    hidden = np.setdiff1d(np.arange(H * K), obs)
    s = supplied[:, obs_cols].reshape(-1)
    comps, logliks, distance = [], [], []
    for system in bank:
        mu, omega = joint_moments(system, jumpoff, H)
        vals, vecs = _eig_floor(omega[np.ix_(obs, obs)])
        d = s - mu[obs]
        rot = vecs.T @ d
        distance.append(float(np.sum(rot**2 / vals)) / len(obs))
        logliks.append(-0.5 * (len(obs) * np.log(2 * np.pi) + np.log(vals).sum()
                               + float(np.sum(rot**2 / vals))))
        omega_rs = omega[np.ix_(hidden, obs)]
        G = ((omega_rs @ vecs) / vals[None, :]) @ vecs.T           # Omega_RS Omega_SS^-1
        mu_c = mu[hidden] + G @ d                                   # conditional mean
        omega_c = omega[np.ix_(hidden, hidden)] - G @ omega_rs.T    # conditional covariance
        cv, cvec = _eig_floor(omega_c)
        comps.append((mu_c, cvec * np.sqrt(cv)))
    logliks = np.asarray(logliks)
    weights = np.exp(logliks - logliks.max())
    weights /= weights.sum()                                        # system probabilities
    assign = rng.choice(len(bank), size=n_paths, p=weights)
    flat = np.empty((n_paths, H * K))
    # Draw remaining cells jointly and restore the supplied cells.
    for b in np.unique(assign):
        sel = np.flatnonzero(assign == b)
        mu_c, factor = comps[b]
        z = rng.standard_normal((len(sel), len(hidden)))
        flat[np.ix_(sel, hidden)] = mu_c[None, :] + z @ factor.T
        flat[np.ix_(sel, obs)] = s[None, :]
    paths = flat.reshape(n_paths, H, K)
    paths[:, :, obs_cols] = supplied[None, :, obs_cols]             # restored exactly
    return paths, {"weights": weights, "mahalanobis_per_dimension": np.asarray(distance)}


def sample_gaussian_law(bank, n_paths, horizon, rng, jumpoff):
    """Unconditional draws from the joint normal, one system per path."""
    K = len(jumpoff)
    assign = rng.integers(0, len(bank), size=n_paths)
    flat = np.empty((n_paths, horizon * K))
    for b in np.unique(assign):
        sel = np.flatnonzero(assign == b)
        mu, omega = joint_moments(bank[b], jumpoff, horizon)
        vals, vecs = _eig_floor(omega)
        z = rng.standard_normal((len(sel), len(mu)))
        flat[sel] = mu[None, :] + z @ (vecs * np.sqrt(vals)).T
    return flat.reshape(n_paths, horizon, K)
