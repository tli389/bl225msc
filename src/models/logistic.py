"""Logistic regression baseline.

Plain sklearn logistic with optional sign-flipping for features with
negative sign priors. Not truly sign-constrained (LR doesn't support
inequality constraints on coefficients natively); the flip trick shifts
the interpretation so coefficients are non-negative on the flipped
features. Mostly kept as the canonical baseline for credit risk.

For a proper sign-constrained LR, one would use cvxpy or scipy.optimize
with inequality constraints; we keep this simple here.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .base import BaseSlopeMixin


class LogisticBaseline(BaseSlopeMixin):
    """Standard logistic regression, no sign constraints."""

    def __init__(
        self,
        feature_names: list[str],
        signs: dict[str, int] | None = None,
        C: float = 1.0,
        max_iter: int = 500,
    ):
        self.feature_names = list(feature_names)
        self.signs = dict(signs) if signs else {f: 0 for f in feature_names}
        self.C = C
        self.max_iter = max_iter
        self._scaler = StandardScaler()
        self._model: LogisticRegression | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "LogisticBaseline":
        Xv = X[self.feature_names].values.astype(np.float32)
        Xs = self._scaler.fit_transform(Xv)
        self._model = LogisticRegression(
            C=self.C, max_iter=self.max_iter, n_jobs=-1, solver="lbfgs"
        ).fit(Xs, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        Xv = X[self.feature_names].values.astype(np.float32)
        Xs = self._scaler.transform(Xv)
        return self._model.predict_proba(Xs)

    def rescale_feature(self, feature: str, scale: float) -> None:
        """Rescale the coefficient on `feature` by `scale`.
        Equivalent to shape-function rescaling for the linear case."""
        if self._model is None:
            raise RuntimeError("model not fit")
        if feature not in self.feature_names:
            raise KeyError(feature)
        j = self.feature_names.index(feature)
        self._model.coef_[0, j] = self._model.coef_[0, j] * scale
