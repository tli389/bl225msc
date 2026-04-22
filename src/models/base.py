"""Model interface (adapter pattern).

Every model in src/models/ implements DisciplinableModel. The discipline
layer (src/discipline) operates on this interface without knowing model
internals. Adding a new architecture is just implementing these methods.
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable
import numpy as np
import pandas as pd


@runtime_checkable
class DisciplinableModel(Protocol):
    """Common interface for monotone-disciplinable satellite models.

    Implementations must set `feature_names` (list of column names in the
    order the model expects) and `signs` (dict feature_name -> {+1, -1, 0}).
    """

    feature_names: list[str]
    signs: dict[str, int]

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "DisciplinableModel":
        """Train on (X, y). X has columns matching feature_names."""
        ...

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return shape (n, 2) class probabilities, index 1 = default."""
        ...

    def measure_feature_slope(
        self,
        X: pd.DataFrame,
        feature: str,
        x_lo: float,
        x_hi: float,
        n_sample: int = 10_000,
        seed: int = 0,
    ) -> float:
        """Finite-difference slope of mean predicted PD w.r.t. `feature`,
        averaged over a random subsample of rows, with all other features
        held at their row values.

        Returns the slope in pp-PD per 1 unit of feature.
        """
        ...

    def rescale_feature(self, feature: str, scale: float) -> None:
        """Rescale the feature's internal shape function by `scale`.
        Mutates the model in place. Preserves monotonicity as long as
        scale > 0.

        Raises NotImplementedError if the model architecture does not
        support per-feature rescaling (e.g., non-additive models).
        """
        ...


# ---------------------------------------------------------------------------
# Default slope measurement — works for any model that returns predict_proba.
# Each concrete model class can inherit from BaseSlopeMixin to get this
# implementation for free.
# ---------------------------------------------------------------------------

class BaseSlopeMixin:
    """Finite-difference slope via predict_proba.  Assumes self has
    `feature_names` and a working `predict_proba`."""

    feature_names: list[str]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray: ...   # type: ignore[empty-body]

    def measure_feature_slope(
        self,
        X: pd.DataFrame,
        feature: str,
        x_lo: float,
        x_hi: float,
        n_sample: int = 10_000,
        seed: int = 0,
    ) -> float:
        if feature not in X.columns:
            raise KeyError(f"{feature} not in X columns")
        sub = X.sample(min(n_sample, len(X)), random_state=seed).reset_index(drop=True)
        X_lo = sub.copy(); X_lo[feature] = x_lo
        X_hi = sub.copy(); X_hi[feature] = x_hi
        p_lo = self.predict_proba(X_lo)[:, 1]
        p_hi = self.predict_proba(X_hi)[:, 1]
        # pp-PD per unit of feature
        return float((p_hi.mean() - p_lo.mean()) * 100.0 / (x_hi - x_lo))
