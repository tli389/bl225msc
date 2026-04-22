"""XGBoost classifier, monotone-optional.

Standard XGBoost with monotone_constraints passed as a tuple of
{+1, -1, 0} per feature. Post-fit rescaling of one feature's
contribution is NOT directly supported by XGBoost (the tree ensemble
doesn't decompose per-feature cleanly) — rescale_feature raises
NotImplementedError. Use this model only for non-IV-disciplined
benchmarks, or accept that IV discipline requires an additive model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from .base import BaseSlopeMixin


class MonotoneXGB(BaseSlopeMixin):
    """XGBoost with optional monotone constraints. Does NOT support
    post-fit IV-discipline rescaling (non-additive architecture)."""

    def __init__(
        self,
        feature_names: list[str],
        signs: dict[str, int] | None = None,
        monotone: bool = True,
        n_estimators: int = 400,
        max_depth: int = 5,
        learning_rate: float = 0.06,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 0,
    ):
        self.feature_names = list(feature_names)
        self.signs = dict(signs) if signs else {f: 0 for f in feature_names}
        self.monotone = monotone
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self._model: XGBClassifier | None = None

    def _mono_tuple(self) -> str | None:
        if not self.monotone:
            return None
        return "(" + ",".join(str(self.signs.get(f, 0))
                              for f in self.feature_names) + ")"

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MonotoneXGB":
        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            tree_method="hist",
            monotone_constraints=self._mono_tuple(),
            n_jobs=-1, verbosity=0,
            random_state=self.random_state,
        ).fit(X[self.feature_names].values.astype(np.float32), y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(
            X[self.feature_names].values.astype(np.float32)
        )

    def rescale_feature(self, feature: str, scale: float) -> None:
        raise NotImplementedError(
            "XGBoost does not support per-feature post-fit rescaling "
            "(non-additive architecture). Use MonotoneEBM or MonotoneNAM "
            "for IV discipline, or apply discipline via training-time "
            "penalty if using XGB."
        )
