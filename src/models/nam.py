"""Monotone Neural Additive Model.

Per-feature MLP with softplus-parameterised non-negative weights (for
sign-constrained features) + tanh-bounded per-feature contribution to
prevent extrapolation blow-up. Additive sum across features + bias →
logit → sigmoid.

Supports IV-discipline rescaling via a per-feature `alpha` buffer
(multiplicative gate, init 1.0). Rescale = set alpha[j] = scale; at
inference, each feature's contribution is multiplied by alpha[j] before
being summed.

Architectural details (from the earlier `nam_monotone.py` that worked):
  - hidden=32 per-feature units (single hidden layer)
  - softplus weight parameterisation for sign ≠ 0
  - tanh output wrapper bounds contribution to (-1, +1) log-odds
  - sign=-1 implemented by flipping the tanh output
  - sign= 0 uses unconstrained linear → ReLU → linear → tanh
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

from .base import BaseSlopeMixin


class _PerFeatureMLP(nn.Module):
    """One hidden layer, softplus-positive weights for sign != 0,
    tanh-bounded output."""

    def __init__(self, hidden: int, sign: int):
        super().__init__()
        self.sign = sign
        self.hidden = hidden
        self.W1 = nn.Parameter(torch.randn(1, hidden) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(hidden))
        self.W2 = nn.Parameter(torch.randn(hidden, 1) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = 1.0 / (self.hidden ** 0.5)
        if self.sign != 0:
            W1 = F.softplus(self.W1) * scale          # all positive
            W2 = F.softplus(self.W2) * scale
            h = F.relu(x @ W1 + self.b1)              # monotone non-decreasing
            out = h @ W2 + self.b2
            return self.sign * torch.tanh(out)        # flip for sign=-1
        else:
            h = F.relu(x @ self.W1 + self.b1)
            return torch.tanh(h @ self.W2 + self.b2)


class _NAM(nn.Module):
    """Internal nn.Module that does the actual forward pass."""

    def __init__(self, feat_signs: list[int], hidden: int):
        super().__init__()
        self.nets = nn.ModuleList([_PerFeatureMLP(hidden, s) for s in feat_signs])
        self.bias = nn.Parameter(torch.zeros(1))
        self.register_buffer("alpha", torch.ones(len(feat_signs)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bias.expand(x.shape[0], 1)
        for i, net in enumerate(self.nets):
            out = out + self.alpha[i] * net(x[:, i:i+1])
        return out.squeeze(-1)


class MonotoneNAM(BaseSlopeMixin):
    """Additive neural model with architectural monotonicity +
    per-feature IV-rescale gate."""

    def __init__(
        self,
        feature_names: list[str],
        signs: dict[str, int] | None = None,
        hidden: int = 32,
        epochs: int = 20,
        batch_size: int = 4096,
        lr: float = 5e-4,
        weight_decay: float = 1e-2,
        random_state: int = 0,
        device: str | None = None,
    ):
        self.feature_names = list(feature_names)
        self.signs = dict(signs) if signs else {f: 0 for f in feature_names}
        self.hidden = hidden
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model: _NAM | None = None
        self._scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "MonotoneNAM":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        Xv = X[self.feature_names].values.astype(np.float32)
        Xs = self._scaler.fit_transform(Xv).astype(np.float32)
        feat_signs = [self.signs.get(f, 0) for f in self.feature_names]

        model = _NAM(feat_signs, self.hidden).to(self.device)
        # init bias at empirical log-odds of base rate
        base = float(np.asarray(y).mean())
        with torch.no_grad():
            model.bias.fill_(float(np.log(max(base, 1e-9) / max(1 - base, 1e-9))))

        loss_fn = nn.BCEWithLogitsLoss()
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr,
                                 weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)

        Xt = torch.from_numpy(Xs).to(self.device)
        yt = torch.from_numpy(np.asarray(y).astype(np.float32)).to(self.device)
        n = len(Xt)
        for _ in range(self.epochs):
            model.train()
            perm = torch.randperm(n, device=self.device)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(model(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            sched.step()

        self._model = model
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model not fit")
        Xv = X[self.feature_names].values.astype(np.float32)
        Xs = self._scaler.transform(Xv).astype(np.float32)
        Xt = torch.from_numpy(Xs).to(self.device)
        self._model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xt), 16384):
                logits = self._model(Xt[i:i + 16384])
                out.append(torch.sigmoid(logits).cpu().numpy())
        p = np.concatenate(out)
        return np.column_stack([1 - p, p])

    def rescale_feature(self, feature: str, scale: float) -> None:
        if self._model is None:
            raise RuntimeError("model not fit")
        if feature not in self.feature_names:
            raise KeyError(feature)
        j = self.feature_names.index(feature)
        with torch.no_grad():
            self._model.alpha[j] = self._model.alpha[j] * scale
