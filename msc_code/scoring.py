"""scoring.py -- the thesis's seven table columns, exactly as specified.

fair CRPS (M(M-1) correction), qWCRPS (J=1000 midpoint levels, US tail map),
energy score (M_E=min(M,300), evenly spaced, fair), variogram (all M, p=0.5),
coverage 80/95, rank PIT averaged over seeds then pooled into PIT-KS.
Formulas match the standalone package (verified against envelope.py and
tail_scores.py); only the code is miniature.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

# tail directions, US and UK (identical to the locked protocols)
TAILS = {"gdp_growth": "lower", "unemployment": "upper",
         "treasury_3m": "two_sided", "treasury_10y": "two_sided",
         "bbb_spread": "upper", "hpi_growth": "lower",
         "cre_growth": "lower", "equity_growth": "lower",
         "real_gdp_growth_annualized_pct": "lower",
         "unemployment_rate_pct": "upper", "bank_rate_pct": "two_sided",
         "gilt_10y_pct": "two_sided", "ig_corporate_spread_pct": "upper",
         "hpi_qoq_growth_pct": "lower", "equity_qoq_growth_pct": "lower",
         "cpi_inflation_yoy_pct": "two_sided"}
J = 1000


def fair_crps(ens, y):
    z = np.sort(ens)
    m = len(z)
    gaps = (2 * np.arange(1, m + 1) - m - 1) * z
    return np.abs(z - y).mean() - gaps.sum() / (m * (m - 1))


def qwcrps(ens, y, tail):
    a = (np.arange(J) + 0.5) / J
    q = np.quantile(ens, a, method="inverted_cdf")
    qs = 2 * (a * np.maximum(y - q, 0) + (1 - a) * np.maximum(q - y, 0))
    w = {"lower": 3 * (1 - a) ** 2, "upper": 3 * a ** 2,
         "two_sided": 3 * (2 * a - 1) ** 2}[tail]
    return (w * qs).sum() / w.sum()


def energy_score(paths_flat, y_flat):
    m = len(paths_flat)
    idx = np.linspace(0, m - 1, min(m, 300), dtype=int)  # floor, as package
    z = paths_flat[idx]
    me = len(z)
    t1 = np.linalg.norm(z - y_flat, axis=1).mean()
    d = np.linalg.norm(z[:, None] - z[None, :], axis=2)
    return t1 - d.sum() / (2 * me * (me - 1))


def variogram_score(paths_flat, y_flat, p=0.5):
    d = paths_flat.shape[1]
    iu = np.triu_indices(d, 1)
    ydiff = np.abs(y_flat[iu[0]] - y_flat[iu[1]]) ** p
    zdiff = np.abs(paths_flat[:, iu[0]] - paths_flat[:, iu[1]]) ** p
    return ((ydiff - zdiff.mean(0)) ** 2).sum() / len(iu[0])


def rank_pit(ens, y):
    m = len(ens)
    return ((ens < y).sum() + 0.5 * (ens == y).sum() + 0.5) / (m + 1)


def score_seed_origin(paths, realised, train_std, features):
    """All per-(seed,origin) quantities for one model. paths: (M,H,K)."""
    H, K = realised.shape
    cells = [(h, j) for h in range(H) for j in range(K)]
    crps = np.mean([fair_crps(paths[:, h, j], realised[h, j]) / train_std[j]
                    for h, j in cells])
    qw = np.mean([qwcrps(paths[:, h, j], realised[h, j],
                         TAILS[features[j]]) / train_std[j]
                  for h, j in cells])
    zf = (paths / train_std).reshape(len(paths), -1)
    yf = (realised / train_std).ravel()
    es, vs = energy_score(zf, yf), variogram_score(zf, yf)
    lo80, hi80 = np.percentile(paths, [10, 90], axis=0)
    lo95, hi95 = np.percentile(paths, [2.5, 97.5], axis=0)
    cov80 = np.mean((realised >= lo80) & (realised <= hi80))
    cov95 = np.mean((realised >= lo95) & (realised <= hi95))
    pits = np.array([rank_pit(paths[:, h, j], realised[h, j])
                     for h, j in cells])
    return dict(crps=crps, qwcrps=qw, energy=es, variogram=vs,
                cov80=cov80, cov95=cov95, pits=pits)


def aggregate(per_seed_origin):
    """Thesis hierarchy: mean over seeds within origin, then over origins;
    PITs averaged across seeds per cell, pooled, then KS vs uniform."""
    out = {}
    for key in ("crps", "qwcrps", "energy", "variogram", "cov80", "cov95"):
        by_origin = [np.mean([s[key] for s in seeds])
                     for seeds in per_seed_origin]
        out[key] = float(np.mean(by_origin))
    pooled = np.concatenate([np.mean([s["pits"] for s in seeds], axis=0)
                             for seeds in per_seed_origin])
    out["pit_ks"] = float(stats.kstest(pooled, "uniform").statistic)
    return out
