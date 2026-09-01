"""Independent AR-t -- the benchmark as the report describes it, one variable at a time:

  x_t = alpha + phi_1 x_{t-1} (+ phi_2 x_{t-2}) + sigma u_t,   u_t ~ t_nu
  1. AR(1) and AR(2) candidates, each fitted by conditional maximum
     likelihood (intercept, AR coefficients, scale, degrees of freedom
     jointly), with 2.1 <= nu <= 30 and the AR polynomial kept inside the
     0.995 stability region through a partial-autocorrelation parameterisation
  2. both candidates scored on the same post-lag-two sample; the retained
     order minimises BIC = -2 loglik + (p + 3) log T
  3. generation: fitted parameters fixed, independent Student-t shocks drawn
     across variables and quarters; in the conditional task the supplied
     column simply replaces the generated one (no cross-variable updating)
Implementation details kept so the fit is exact: the series is standardised
before optimisation, L-BFGS-B is started from three degrees-of-freedom values,
and the best converged start is kept.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

MIN_SCALE = float(np.nextafter(0.001, np.inf))


def _radius(phi):
    if len(phi) == 1:
        return abs(float(phi[0]))
    companion = np.array([[phi[0], phi[1]], [1.0, 0.0]])
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def _coef_from_pacf(params, order, max_r):
    """Durbin-Levinson: |kappa| < 1 is exactly the stable region; then cap."""
    kappa = np.tanh(np.asarray(params[:order], float))
    phi = kappa if order == 1 else np.array([kappa[0] * (1.0 - kappa[1]), kappa[1]])
    return phi * max_r ** np.arange(1, order + 1)


def _pacf_from_coef(phi, max_r):
    if len(phi) == 1:
        kappa = np.array([phi[0] / max_r])
    else:
        second = phi[1] / max_r**2
        kappa = np.array([(phi[0] / max_r) / max(1.0 - second, 1e-8), second])
    return np.arctanh(np.clip(kappa, -0.995, 0.995))


def _project_stable(phi, max_r):
    if _radius(phi) < max_r:
        return phi.copy()
    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _radius(mid * phi) < max_r else (lo, mid)
    return lo * phi


def _loglik(resid, scale, df):
    const = gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(df * np.pi) - np.log(scale)
    return float(len(resid) * const - 0.5 * (df + 1) * np.sum(np.log1p((resid / scale) ** 2 / df)))


def fit_candidate(series, order, max_r=0.995, min_df=2.1, max_df=30.0, hold_back=2):
    """Conditional MLE of one AR(p)-t candidate on the common sample."""
    v = np.asarray(series, float)
    target_raw = v[hold_back:]
    lagged_raw = np.column_stack([v[hold_back - lag:len(v) - lag] for lag in range(1, order + 1)])
    centre, norm = float(v.mean()), float(v.std(ddof=0)) or 1.0
    if norm <= 1e-8:
        norm = 1.0
    target, lagged = (target_raw - centre) / norm, (lagged_raw - centre) / norm

    # starting point: OLS pushed into the stable region
    ols, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(target)), lagged]), target, rcond=None)
    phi0 = _project_stable(ols[1:], max_r * (1.0 - 1e-8))
    alpha0 = float(np.mean(target - lagged @ phi0))
    rms = float(np.sqrt(np.mean((norm * (target - alpha0 - lagged @ phi0)) ** 2)))
    df0 = min(max(5.0, min_df), max_df)
    scale0 = max(rms * np.sqrt((df0 - 2.0) / df0), MIN_SCALE * 1.05)
    start = np.concatenate([[alpha0], _pacf_from_coef(phi0, max_r), [np.log(scale0), df0]])
    scale_upper = max(MIN_SCALE * 100.0, float(np.ptp(v)) * 100.0, norm * 100.0)
    bounds = ([(None, None)] + [(-6.0, 6.0)] * order
              + [(np.log(MIN_SCALE), np.log(scale_upper)), (min_df, max_df)])

    def unpack(p):
        phi = _coef_from_pacf(p[1:1 + order], order, max_r)
        return phi, float(np.exp(p[-2])), float(p[-1]), norm * (target - p[0] - lagged @ phi)

    def objective(p):
        _, scale, df, resid = unpack(p)
        ll = _loglik(resid, scale, df)
        return -ll if np.isfinite(ll) else 1e100

    attempts = []
    for df_start in dict.fromkeys((df0, min_df + 0.5, max_df)):
        s = start.copy()
        s[-1] = float(np.clip(df_start, min_df, max_df))
        attempts.append(minimize(objective, s, method="L-BFGS-B", bounds=bounds,
                                 options={"ftol": 1e-10, "gtol": 1e-6, "maxiter": 500, "maxls": 50}))
    ok = [r for r in attempts if r.success and np.isfinite(r.fun) and np.isfinite(r.x).all()]
    if not ok:
        return None
    best = min(ok, key=lambda r: float(r.fun))
    phi, scale, df, resid = unpack(best.x)
    intercept = centre * (1.0 - float(np.sum(phi))) + norm * float(best.x[0])
    bic = -2.0 * _loglik(resid, scale, df) + (order + 3) * np.log(len(target_raw))
    return {"order": order, "coef": phi, "intercept": intercept, "scale": scale, "df": df, "bic": bic}


def fit_ar_t(values, **kw):
    """values: (T, K).  Returns per-variable parameters and the last two states."""
    values = np.asarray(values, float)
    K = values.shape[1]
    coef, intercept, scale, df, order = np.zeros((K, 2)), np.zeros(K), np.zeros(K), np.zeros(K), np.zeros(K, int)
    for k in range(K):
        cands = [c for c in (fit_candidate(values[:, k], p, **kw) for p in (1, 2)) if c is not None]
        best = min(cands, key=lambda c: (c["bic"], c["order"]))
        coef[k, :best["order"]] = best["coef"]
        intercept[k], scale[k], df[k], order[k] = best["intercept"], best["scale"], best["df"], best["order"]
    return {"coef": coef, "intercept": intercept, "scale": scale, "df": df,
            "order": order, "tail": values[-2:].copy()}


def sample_ar_t(fit, n_paths, horizon, rng, jumpoff=None):
    """Independent t shocks; the jump-off replaces the last training state."""
    tail = fit["tail"].copy()
    if jumpoff is not None:
        tail[-1] = jumpoff
    K = fit["coef"].shape[0]
    paths = np.empty((n_paths, horizon, K))
    for k in range(K):
        prev, prev2 = np.full(n_paths, tail[-1, k]), np.full(n_paths, tail[-2, k])
        shocks = rng.standard_t(fit["df"][k], size=(n_paths, horizon)) * fit["scale"][k]
        for h in range(horizon):
            state = fit["intercept"][k] + fit["coef"][k, 0] * prev + fit["coef"][k, 1] * prev2 + shocks[:, h]
            paths[:, h, k] = state
            prev2, prev = prev, state
    return paths
