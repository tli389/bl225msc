"""Monotone Explainable Boosting Machine (Zhou-style primary satellite).

EBM with per-feature monotone sign constraints. Supports IV-discipline
rescaling natively by multiplying the feature's stored shape-function
term_scores_ by a scalar (preserves monotonicity since scale > 0).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier

from .base import BaseSlopeMixin


class MonotoneEBM(BaseSlopeMixin):
    """EBM with sign-constrained main-effect shape functions plus
    automatically-selected pairwise interactions."""

    def __init__(
        self,
        feature_names: list[str],
        signs: dict[str, int] | None = None,
        interactions: int = 10,
        max_bins: int = 256,
        outer_bags: int = 8,
        learning_rate: float = 0.05,
        min_samples_leaf: int = 20,
        random_state: int = 0,
    ):
        self.feature_names = list(feature_names)
        self.signs = dict(signs) if signs else {f: 0 for f in feature_names}
        self.interactions = interactions
        self.max_bins = max_bins
        self.outer_bags = outer_bags
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self._model: ExplainableBoostingClassifier | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MonotoneEBM":
        self._model = ExplainableBoostingClassifier(
            feature_names=self.feature_names,
            monotone_constraints=[self.signs.get(f, 0) for f in self.feature_names],
            interactions=self.interactions,
            max_bins=self.max_bins,
            outer_bags=self.outer_bags,
            learning_rate=self.learning_rate,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(X[self.feature_names].values.astype(np.float32), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(
            X[self.feature_names].values.astype(np.float32)
        )

    # ---- IV discipline support ----

    def _find_main_effect_idx(self, feature: str) -> int:
        """Locate the main-effect term for `feature` in term_features_."""
        target = self.feature_names.index(feature)
        for i, t in enumerate(self._model.term_features_):
            if len(t) == 1 and t[0] == target:
                return i
        raise ValueError(f"no main-effect term for {feature}")

    def rescale_feature(self, feature: str, scale: float) -> None:
        """Multiply the feature's shape function by `scale`. Preserves
        monotonicity if scale > 0."""
        if self._model is None:
            raise RuntimeError("model not fit")
        idx = self._find_main_effect_idx(feature)
        self._model.term_scores_[idx] = self._model.term_scores_[idx] * scale

    def get_shape_function(self, feature: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (bin_centres, score_per_bin) for plotting the learned
        shape function of a main effect."""
        if self._model is None:
            raise RuntimeError("model not fit")
        idx = self._find_main_effect_idx(feature)
        feat_idx = self.feature_names.index(feature)
        edges = np.asarray(self._model.bins_[feat_idx][0])
        scores = np.asarray(self._model.term_scores_[idx])[1:-1]
        if len(edges) > 1:
            centres = 0.5 * (edges[:-1] + edges[1:])
        else:
            centres = edges
        m = min(len(centres), len(scores))
        return centres[:m], scores[:m]
