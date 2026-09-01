"""Student-t residual-block bridge -- the report's conditional adapter.

Implemented from the description in the report's Proposed Approach chapter
with the same constants (10,000 particles, 25 systems, tau = 0.75, nu = 20,
4-quarter blocks).  One difference from the runs behind the report's
Student-t rows: those runs added an adaptive likelihood-tempering step (with
a rejuvenation move) whenever the weights concentrated within a stage; this
bridge handles weight concentration by resampling alone.  The two agree to
within SMC noise -- check_benchmarks.py and check_narrative.py show
the Student-t rows landing within a few thousandths of the report's.

Report step -> code:
  * particles spread uniformly over the bootstrap systems, each with its own
    residual pool                                        (clean_bridge_sample)
  * candidate centres = every 4-quarter block of the pool, shrunk by
    sqrt(1 - tau^2)                                      (_block_centres)
  * t-noise scale S_eta = ((nu-2)/nu) tau^2 (I_L (x) Sigma)      (_System)
  * H maps the stage's L*K innovations to the q supplied cells   (_projection)
  * u_r = d - H(centre_r), delta_r = u_r' S_obs^{-1} u_r, S_obs = H S_eta H',
    L_r = t_nu density of u_r (normalising constant included)   (stage step 1)
  * weight x= average of L_r over blocks; one block drawn ~ L_r (steps 2, 3)
  * eta | H eta = u_r ~ t_{nu+q}(G u_r, ((nu+delta_r)/(nu+q)) S_c),
    G = S_eta H' S_obs^{-1}, S_c = S_eta - G H S_eta  (Ding, 2016)  (step 3)
  * systematic resampling when ESS < 0.5 N, performed before the stage's
    draws so duplicated particles receive independent draws       (step 2)
  * 1,000 paths harvested proportional to the final weights; supplied cells
    clamped                                                  (end of function)
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln, logsumexp


def _block_centres(residuals, length, tau):
    n = len(residuals) - length + 1
    blocks = np.stack([residuals[s:s + length] for s in range(n)])   # (n, L, K)
    mean = residuals.mean(axis=0)
    return (mean + np.sqrt(1.0 - tau**2) * (blocks - mean)).reshape(n, -1)


def _projection(coef, length, obs_idx):
    """Rows of the innovation->state map that land on the supplied cells."""
    k = coef.shape[0]
    powers = [np.eye(k)]
    for _ in range(1, length):
        powers.append(coef @ powers[-1])
    full = np.zeros((length * k, length * k))
    for later in range(length):
        for earlier in range(later + 1):
            full[later * k:(later + 1) * k, earlier * k:(earlier + 1) * k] = \
                powers[later - earlier]
    rows = [s * k + f for s in range(length) for f in obs_idx]
    return full[rows]                                                # (q, L*K)


def _inv_logdet(m):
    m = 0.5 * (m + m.T)
    vals, vecs = np.linalg.eigh(m)
    vals = np.maximum(vals, max(vals.max(), 1.0) * 1e-10)
    return (vecs / vals) @ vecs.T, float(np.log(vals).sum())


def _root(m):
    m = 0.5 * (m + m.T)
    vals, vecs = np.linalg.eigh(m)
    keep = vals > max(vals.max(), 1.0) * 1e-10
    return vecs[:, keep] * np.sqrt(vals[keep])


def _systematic(weights, rng):
    positions = (rng.random() + np.arange(len(weights))) / len(weights)
    cum = np.cumsum(weights)
    cum[-1] = 1.0
    return np.searchsorted(cum, positions, side="right")


class _System:
    """Per-system, per-stage-length precomputation (pure report algebra)."""

    def __init__(self, dyn, length, obs_idx, tau, nu):
        self.A = np.asarray(dyn.coef, float)
        self.c = np.asarray(dyn.intercept, float)
        k = self.A.shape[0]
        sigma = np.asarray(dyn.residual_cov, float)
        self.centres = _block_centres(np.asarray(dyn.residuals, float), length, tau)
        self.H = _projection(self.A, length, obs_idx)
        s_eta = ((nu - 2.0) / nu) * tau**2 * np.kron(np.eye(length), sigma)
        s_obs = self.H @ s_eta @ self.H.T
        self.prec, self.logdet = _inv_logdet(s_obs)
        self.G = s_eta @ self.H.T @ self.prec
        self.root = _root(s_eta - self.G @ self.H @ s_eta)
        self.centre_obs = self.centres @ self.H.T                    # (n_b, q)
        q = self.H.shape[0]
        self.const = (gammaln((nu + q) / 2) - gammaln(nu / 2)
                      - 0.5 * (q * np.log(nu * np.pi) + self.logdet))
        self.q, self.k, self.length = q, k, length

    def zero_shock_observed(self, states, obs_idx):
        cur = states.copy()
        out = np.empty((len(states), self.length, len(obs_idx)))
        for step in range(self.length):
            cur = cur @ self.A.T + self.c
            out[:, step] = cur[:, obs_idx]
        return out.reshape(len(states), -1)


def clean_bridge_sample(generator, observed_values, obs_idx, *, n_paths,
                        n_particles, rng, jumpoff, block_length=4, tau=0.75,
                        nu=20.0, ess_ratio=0.5):
    """observed_values: (horizon, q) supplied cells; obs_idx: their columns."""
    systems = list(getattr(generator, "draws_", generator))   # a bank, or an object holding one
    obs_idx = np.asarray(obs_idx, int)
    horizon = len(observed_values)
    k = len(jumpoff)
    n = n_particles

    assign = rng.integers(0, len(systems), size=n)
    states = np.repeat(np.asarray(jumpoff, float)[None, :], n, axis=0)
    paths = np.empty((n, horizon, k))
    weights = np.full(n, 1.0 / n)
    ancestors = np.arange(n)                     # diagnostics only
    ess_log, block_shares, resamples = [], [], 0

    start = 0
    while start < horizon:
        length = min(block_length, horizon - start)
        target = observed_values[start:start + length].reshape(-1)
        built = {b: _System(systems[b], length, obs_idx, tau, nu)
                 for b in np.unique(assign)}
        n_b = next(iter(built.values())).centres.shape[0]
        q = len(target)

        # 1. block log-likelihoods for every particle x candidate block
        loglik = np.empty((n, n_b))
        diffs = np.empty((n, n_b, q))
        for b, sysb in built.items():
            rows = np.flatnonzero(assign == b)
            base = sysb.zero_shock_observed(states[rows], obs_idx)      # (m, q)
            d = target[None, None, :] - base[:, None, :] - sysb.centre_obs[None]
            quad = np.einsum("...i,ij,...j->...", d, sysb.prec, d, optimize=True)
            loglik[rows] = sysb.const - 0.5 * (nu + q) * np.log1p(quad / nu)
            diffs[rows] = d
        logpred = logsumexp(loglik, axis=1) - np.log(n_b)   # log average L_r

        # 2. weight update, then resample BEFORE drawing (fully adapted step)
        logw = np.log(np.maximum(weights, np.finfo(float).tiny)) + logpred
        weights = np.exp(logw - logsumexp(logw))
        weights /= weights.sum()
        ess = 1.0 / np.sum(weights**2)
        ess_log.append(ess / n)
        if ess < ess_ratio * n:
            sel = _systematic(weights, rng)
            states, paths, assign = states[sel], paths[sel], assign[sel]
            loglik, diffs, ancestors = loglik[sel], diffs[sel], ancestors[sel]
            resamples += 1
            weights = np.full(n, 1.0 / n)

        # 3. per particle: choose a block ~ L_r, draw eta | H eta = u_r, propagate
        chosen_all = np.empty(n, int)
        for b, sysb in built.items():
            rows = np.flatnonzero(assign == b)
            if not len(rows):
                continue
            ll = loglik[rows]
            prob = np.exp(ll - ll.max(axis=1, keepdims=True))
            cum = np.cumsum(prob, axis=1)
            u = rng.random(len(rows)) * cum[:, -1]
            chosen = np.sum(cum < u[:, None], axis=1)
            chosen = np.minimum(chosen, n_b - 1)
            chosen_all[rows] = chosen
            d_sel = diffs[rows, chosen]                                   # (m, q)
            delta = np.einsum("ij,jk,ik->i", d_sel, sysb.prec, d_sel)
            mean = sysb.centres[chosen] + d_sel @ sysb.G.T
            z = rng.standard_normal((len(rows), sysb.root.shape[1])) @ sysb.root.T
            scale = np.sqrt((nu + delta) / rng.chisquare(nu + q, size=len(rows)))
            innov = (mean + z * scale[:, None]).reshape(len(rows), length, k)
            cur = states[rows]
            for off in range(length):
                cur = cur @ sysb.A.T + sysb.c + innov[:, off]
                cur[:, obs_idx] = observed_values[start + off]
                paths[rows, start + off] = cur
            states[rows] = cur
        share = np.bincount(assign * n_b + chosen_all, weights=weights)
        share = share[share > 0]
        block_shares.append((1.0 / np.sum(share**2), float(share.max())))
        start += length

    pick = rng.choice(n, size=n_paths, replace=True, p=weights)
    out = paths[pick].copy()
    out[:, :, obs_idx] = observed_values[None]
    comp = np.bincount(assign, weights=weights, minlength=len(systems))
    comp = comp[comp > 0]
    anc = np.bincount(ancestors, weights=weights, minlength=n)
    return out, {"ess_ratio": np.asarray(ess_log),
                 "final_ess_ratio": float(1.0 / np.sum(weights**2) / n),
                 "systems_alive": int(len(comp)),
                 "effective_components": float(1.0 / np.sum(comp**2)),
                 "maximum_component_share": float(comp.max()),
                 "effective_selected_blocks": np.asarray([e for e, _ in block_shares]),
                 "maximum_selected_block_share": np.asarray([m for _, m in block_shares]),
                 "unique_ancestor_ratio": float(np.mean(anc > 0)),
                 "resampling_count": int(resamples)}
