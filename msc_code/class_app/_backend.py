"""Adapt local VAR fits and conditional samplers to the CLASS runners."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SOURCE = Path(__file__).resolve().parent.parent / "src"
if str(_SOURCE) not in sys.path:
    sys.path.insert(0, str(_SOURCE))
from bvar import fit_minnesota_bvar                                 # noqa: E402
from gaussian_adapter import complete                               # noqa: E402
from gibvar import fit_gibvar                                       # noqa: E402
from student_t import clean_bridge_sample                           # noqa: E402


def stable_seed(base_seed, *parts):
    key = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return int(base_seed + int.from_bytes(digest, "little") % 100_000)


def clip_to_bounds(paths, features, bounds=None):
    for index, feature in enumerate(features):
        if bounds and feature in bounds:
            lo, hi = bounds[feature]
            paths[..., index] = np.clip(paths[..., index], lo, hi)
    return paths


class GIBVAR:
    """Fit the configured bank of graph-weighted VAR systems."""

    def __init__(self, n_dynamics=25, block_length=4, expert_weight=0.35, max_parents=3,
                 max_spectral_radius=0.995, random_state=10_000, path_solver="lars", *,
                 features, expert_edges, bounds=None, force_expert_edges=False):
        if path_solver != "lars" or force_expert_edges:
            raise ValueError("src/gibvar.py implements the lars path with soft graph guidance only")
        self.n_dynamics, self.block_length = int(n_dynamics), int(block_length)
        self.expert_weight, self.max_parents = float(expert_weight), int(max_parents)
        self.max_spectral_radius, self.random_state = float(max_spectral_radius), int(random_state)
        self.features, self.expert_edges, self.bounds = tuple(features), dict(expert_edges), bounds
        self.draws_ = self.dynamics_ = self.jumpoff_ = None

    def fit(self, history):
        values = history.loc[:, list(self.features)].to_numpy(dtype=float)
        self.draws_ = fit_gibvar(values, list(self.features), self.expert_edges, self.random_state,
                                 n_systems=self.n_dynamics, block=self.block_length,
                                 weight=self.expert_weight, max_parents=self.max_parents,
                                 max_radius=self.max_spectral_radius)
        self.dynamics_, self.jumpoff_ = self.draws_[0], values[-1].copy()
        return self


class MinnesotaPosteriorVARGenerator:
    """Fit the fixed-prior BVAR with semi-conjugate Gibbs sampling."""

    def __init__(self, n_dynamics=100, delta=0.9, own_scale=0.2, cross_scale=0.1,
                 intercept_scale=10.0, covariance_prior_df=None, burn_in=150, thin=3,
                 max_spectral_radius=0.995, max_sampling_iterations=20000, seed=211, *,
                 features, bounds=None, own_lag_means=None):
        if covariance_prior_df is not None:
            raise ValueError("src/bvar.py uses the K + 2 inverse-Wishart prior only")
        self.n_dynamics, self.delta, self.own_scale = int(n_dynamics), float(delta), float(own_scale)
        self.cross_scale, self.intercept_scale = float(cross_scale), float(intercept_scale)
        self.burn_in, self.thin, self.max_radius = int(burn_in), int(thin), float(max_spectral_radius)
        self.max_iter, self.seed = int(max_sampling_iterations), int(seed)
        self.features, self.bounds, self.own_lag_means = tuple(features), bounds, own_lag_means
        self.draws_ = self.dynamics_ = None

    def fit(self, history):
        values = history.loc[:, list(self.features)].to_numpy(dtype=float)
        own = (None if self.own_lag_means is None
               else [self.own_lag_means.get(f, self.delta) for f in self.features])
        self.draws_ = fit_minnesota_bvar(values, self.seed, own_lag_means=own, delta=self.delta,
                                         own_sd=self.own_scale, cross_sd=self.cross_scale,
                                         intercept_sd=self.intercept_scale, n_draws=self.n_dynamics,
                                         burn_in=self.burn_in, thin=self.thin,
                                         max_radius=self.max_radius, max_iter=self.max_iter)
        self.dynamics_ = self.draws_[0]
        return self


@dataclass(frozen=True)
class ConditionalMixtureDiagnostics:
    component_count: int
    effective_components: float
    maximum_weight: float
    minimum_mahalanobis_per_dimension: float
    posterior_weights: np.ndarray


def conditional_mixture_sample(generator, supplied_path, observed_features, n_paths, rng, jumpoff, *,
                               features=None, bounds=None, apply_hidden_bounds=True):
    features = tuple(generator.features if features is None else features)
    obs_cols = [features.index(name) for name in observed_features]
    # Condition each fitted Gaussian system and reweight the mixture.
    paths, info = complete(list(generator.draws_), np.asarray(supplied_path, float), obs_cols,
                           n_paths, rng, np.asarray(jumpoff, float))
    if apply_hidden_bounds:
        paths = clip_to_bounds(paths, features, bounds)
        paths[:, :, obs_cols] = np.asarray(supplied_path, float)[None, :, obs_cols]
    weights = info["weights"]
    return paths, ConditionalMixtureDiagnostics(
        component_count=len(weights), effective_components=float(1.0 / np.sum(weights**2)),
        maximum_weight=float(weights.max()),
        minimum_mahalanobis_per_dimension=float(info["mahalanobis_per_dimension"].min()),
        posterior_weights=weights)


@dataclass(frozen=True)
class BlockBridgeDiagnostics:
    ess_ratio: np.ndarray
    minimum_ess_ratio: float
    effective_components: float
    maximum_component_share: float
    effective_selected_blocks: np.ndarray
    maximum_selected_block_share: np.ndarray
    unique_ancestor_ratio: float
    resampling_count: int
    tempering_enabled: bool = False


def block_bridge_conditional_sample(generator, supplied_path, observed_features, *, n_paths, n_particles,
                                    rng, jumpoff, block_length=4, tau=0.75, kernel="student_t",
                                    degrees_of_freedom=20.0, ess_resample_ratio=0.5,
                                    apply_hidden_bounds=True, features=None, bounds=None, **_tempering):
    """Use the resampling-only bridge; tempering and rejuvenation options are ignored."""
    if kernel != "student_t":
        raise ValueError("src/student_t.py implements the Student-t kernel only")
    features = tuple(generator.features if features is None else features)
    obs_cols = [features.index(name) for name in observed_features]
    supplied = np.asarray(supplied_path, float)
    observed = supplied if supplied.shape[1] == len(obs_cols) else supplied[:, obs_cols]
    # Pass the shock-law settings to the local block sampler.
    paths, diag = clean_bridge_sample(list(generator.draws_), observed, obs_cols, n_paths=n_paths,
                                      n_particles=n_particles, rng=rng, jumpoff=np.asarray(jumpoff, float),
                                      block_length=block_length, tau=tau, nu=degrees_of_freedom,
                                      ess_ratio=ess_resample_ratio)
    if apply_hidden_bounds:
        paths = clip_to_bounds(paths, features, bounds)
        paths[:, :, obs_cols] = observed[None, :, :]
    return paths, BlockBridgeDiagnostics(
        ess_ratio=diag["ess_ratio"], minimum_ess_ratio=float(diag["ess_ratio"].min()),
        effective_components=diag["effective_components"],
        maximum_component_share=diag["maximum_component_share"],
        effective_selected_blocks=diag["effective_selected_blocks"],
        maximum_selected_block_share=diag["maximum_selected_block_share"],
        unique_ancestor_ratio=diag["unique_ancestor_ratio"],
        resampling_count=diag["resampling_count"])
