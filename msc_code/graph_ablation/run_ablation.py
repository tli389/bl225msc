from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT / "src"), str(ROOT / "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)
import data as D                                                    # noqa: E402
import gibvar as G                                                  # noqa: E402
import results as RES                                               # noqa: E402
import scoring as SC                                                # noqa: E402
from sklearn.linear_model import Lasso, lars_path                  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit                # noqa: E402

HORIZON, N_PATHS, SEEDS, SYSTEMS = 12, 1000, (7, 19, 43), 25
VARIANTS = ("full_gibvar", "equal_penalty", "forced_graph")
LABEL = {"full_gibvar": "Soft graph guidance", "equal_penalty": "Equal penalties",
         "forced_graph": "Forced preferred edges"}
WEIGHT = {"full_gibvar": 0.35, "equal_penalty": 1.0}
PAIRS = (("full_gibvar", "equal_penalty"), ("forced_graph", "equal_penalty"),
         ("forced_graph", "full_gibvar"))
METRICS = {"crps": "crps", "qwcrps": "qwcrps", "energy": "energy_score",
           "variogram": "variogram_score", "cov80": "coverage80",
           "cov95": "coverage95", "width80": "width80", "width95": "width95"}
PROPER = ("crps", "qwcrps", "energy_score", "variogram_score")


# ------------------------------------------------ forced variant, one equation
def _partial_out_forced(x, y, target, forced):
    """Standardise, project out [1, own lag, forced lags], scale the rest."""
    n, K = x.shape
    unpen = np.concatenate(([target], forced))
    optional = np.array([j for j in range(K) if j != target and j not in forced], int)
    mean = x.mean(0)
    scale = np.where(x.std(0) > 1e-8, x.std(0), 1.0)
    z = (x - mean) / scale
    base = np.column_stack([np.ones(n), z[:, unpen]])
    y_proj, *_ = np.linalg.lstsq(base, y, rcond=None)
    x_proj, *_ = np.linalg.lstsq(base, z[:, optional], rcond=None)
    x_res = z[:, optional] - base @ x_proj
    res_scale = np.where(x_res.std(0) > 1e-8, x_res.std(0), 1.0)
    return dict(unpen=unpen, optional=optional, mean=mean, scale=scale, y_proj=y_proj,
                x_proj=x_proj, res_scale=res_scale, design=x_res / res_scale,
                y_res=y - base @ y_proj)


def _apply_forced(fit, x):
    z = (x - fit["mean"]) / fit["scale"]
    base = np.column_stack([np.ones(len(x)), z[:, fit["unpen"]]])
    x_res = z[:, fit["optional"]] - base @ fit["x_proj"]
    return x_res / fit["res_scale"], base @ fit["y_proj"]


def _select_ratio_forced(x, y, target, forced):
    """gibvar._select_ratio with the forced lags in the unpenalised block."""
    ratios = np.geomspace(1.0, 1e-3, 60)
    losses = np.zeros(len(ratios))
    n_splits = min(5, max(2, len(x) // 12))
    folds = list(TimeSeriesSplit(n_splits=n_splits).split(x))
    for tr, va in folds:
        fit = _partial_out_forced(x[tr], y[tr], target, forced)
        design_va, baseline = _apply_forced(fit, x[va])
        alpha_max = float(np.max(np.abs(fit["design"].T @ fit["y_res"])) / max(len(tr), 1))
        if alpha_max <= 1e-12 or np.std(fit["y_res"]) <= 1e-10:
            pred = np.repeat(baseline[:, None], len(ratios), axis=1)
        else:
            alphas = np.maximum(alpha_max * ratios, 1e-12)
            path_alphas, _, path_coefs = lars_path(fit["design"], fit["y_res"],
                                                   method="lasso", alpha_min=0.0)
            order = np.argsort(path_alphas)
            uniq, idx = np.unique(path_alphas[order], return_index=True)
            ordered = path_coefs[:, order][:, idx]
            coefs = np.vstack([np.interp(alphas, uniq, row) for row in ordered])
            pred = baseline[:, None] + design_va @ coefs
        losses += np.mean((y[va, None] - pred) ** 2, axis=0)
    return float(ratios[int(np.argmin(losses / len(folds)))])


def fit_forced_equation(x, y, target, forced, max_parents=3):
    """gibvar._fit_equation with the preferred lags always kept (unpenalised)."""
    n, K = x.shape
    fit = _partial_out_forced(x, y, target, forced)
    gamma = np.zeros(len(fit["optional"]))
    slots = max_parents - len(forced)
    if slots > 0 and len(fit["optional"]) and np.std(fit["y_res"]) > 1e-10:
        ratio = _select_ratio_forced(x, y, target, forced)
        alpha_max = float(np.max(np.abs(fit["design"].T @ fit["y_res"])) / max(n, 1))
        lasso = Lasso(alpha=max(alpha_max * ratio, 1e-12), fit_intercept=False,
                      max_iter=100_000).fit(fit["design"], fit["y_res"])
        ranked = np.argsort(np.abs(lasso.coef_))[::-1]
        keep = [int(j) for j in ranked if abs(float(lasso.coef_[j])) > 1e-12][:slots]
        if keep:
            refit = Lasso(alpha=float(lasso.alpha), fit_intercept=False,
                          max_iter=100_000).fit(fit["design"][:, keep], fit["y_res"])
            gamma[keep] = refit.coef_ / fit["res_scale"][keep]
    z = (x - fit["mean"]) / fit["scale"]
    base = np.column_stack([np.ones(n), z[:, fit["unpen"]]])
    base_coef, *_ = np.linalg.lstsq(base, y - z[:, fit["optional"]] @ gamma, rcond=None)
    beta_z = np.zeros(K)
    beta_z[fit["optional"]] = gamma
    beta_z[fit["unpen"]] = base_coef[1:]
    beta = beta_z / fit["scale"]
    return beta, float(base_coef[0]) - float(beta @ fit["mean"])


def fit_forced_system(x, y, features, graph, max_parents=3, max_radius=0.995):
    """gibvar.fit_system with fit_forced_equation for every equation."""
    K = x.shape[1]
    coef, intercept = np.zeros((K, K)), np.zeros(K)
    for k in range(K):
        preferred = set(graph.get(features[k], ()))
        forced = np.array([j for j in range(K) if j != k and features[j] in preferred], int)
        coef[k], intercept[k] = fit_forced_equation(x, y[:, k], k, forced, max_parents)
    radius = float(np.max(np.abs(np.linalg.eigvals(coef))))
    coef = coef * (1.0 if radius <= max_radius else max_radius / radius)
    intercept = y.mean(0) - x.mean(0) @ coef.T
    residuals = y - (x @ coef.T + intercept)
    cov = G._regularised_cov(residuals)
    shift = residuals.mean(0)
    return G.System(intercept + shift, coef, cov, residuals - shift)


# ----------------------------------------------------- three matched banks
def fit_banks(values, features, graph, seed):
    """gibvar.fit_gibvar with the three variants fitted on every resample."""
    values = np.asarray(values, float)
    x, y = values[:-1], values[1:]
    rng = np.random.default_rng(seed + len(values))
    banks = {v: [] for v in VARIANTS}
    for _ in range(SYSTEMS):
        idx = G._block_pair_indices(len(x), 4, rng)                # shared draw
        for v in VARIANTS:
            if v == "forced_graph":
                cand = fit_forced_system(values[idx], values[idx + 1], features, graph)
            else:
                cand = G.fit_system(values[idx], values[idx + 1], features, graph,
                                    weight=WEIGHT[v])
            r = y - (x @ cand.coef.T + cand.intercept)                # chronological
            cov = G._regularised_cov(r)
            shift = r.mean(0)
            banks[v].append(G.System(cand.intercept + shift, cand.coef, cov, r - shift))
    return banks


# ------------------------------------------------------------------ tables
def summarise(seed_scores, cell_pits, names, crisis):
    """Seed-averaged origin scores -> overall / crisis / calm tables."""
    by_origin = (seed_scores.groupby(["origin", "method"], sort=False)[list(METRICS.values())]
                 .mean().reset_index())
    by_origin["crisis"] = by_origin.origin.map(dict(zip(names, crisis)))
    rows = []
    for regime in ("overall", "crisis", "calm"):
        sel = by_origin if regime == "overall" else by_origin[by_origin.crisis.eq(regime == "crisis")]
        if sel.empty:
            continue
        for v in ("equal_penalty", "full_gibvar", "forced_graph"):
            sub = sel[sel.method.eq(v)]
            pits = cell_pits[cell_pits.method.eq(v) & cell_pits.origin.isin(sub.origin)].pit.to_numpy()
            rows.append({"domain": "us_class8", "country": "US", "task": "unconditional",
                         "method": v, "model": LABEL[v], "scoring_policy": "raw",
                         "regime": regime, "n_origins": len(sub), "n_pit_cells": len(pits),
                         **{m: float(sub[m].mean()) for m in METRICS.values()},
                         "pit_mean": float(pits.mean()),
                         "pit_ks": float(stats.kstest(pits, "uniform").statistic),
                         "pit_cvm": float(stats.cramervonmises(pits, "uniform").statistic)})
    return by_origin, pd.DataFrame(rows)


def differences(by_origin, names, crisis):
    """Matched differences of the seed-averaged origin scores, with intervals."""
    rows = []
    for left, right in PAIRS:
        a = by_origin[by_origin.method.eq(left)].set_index("origin").loc[names]
        b = by_origin[by_origin.method.eq(right)].set_index("origin").loc[names]
        for metric in PROPER:
            delta = a[metric].to_numpy() - b[metric].to_numpy()
            for regime in ("overall", "calm", "crisis"):
                mask = np.ones(len(delta), bool) if regime == "overall" else crisis == (regime == "crisis")
                if not mask.any():
                    continue
                seed = RES.stable_seed(7, "paired", "raw", "unconditional", left, right, metric, regime)
                lo, hi = RES.block_bootstrap_interval(delta, crisis, regime, seed)
                rows.append({"left_variant": left, "left_label": LABEL[left],
                             "right_variant": right, "right_label": LABEL[right],
                             "metric": metric, "regime": regime, "n_origins": int(mask.sum()),
                             "mean_difference_left_minus_right": float(delta[mask].mean()),
                             "median_difference_left_minus_right": float(np.median(delta[mask])),
                             "fraction_origins_left_lower": float((delta[mask] < 0).mean()),
                             "interpretation": "negative_difference_favors_left_variant",
                             "ci95_low": lo, "ci95_high": hi})
    return pd.DataFrame(rows)


def compare_with_reference(out):
    """Every numeric entry of the four tables against reference/."""
    rows = []
    for name in ("table.csv", "crisis_table.csv", "calm_table.csv", "matched_differences.csv"):
        keys = (["left_variant", "right_variant", "metric", "regime"]
                if name == "matched_differences.csv" else ["method", "regime"])
        ref = pd.read_csv(HERE / "reference" / name)
        new = pd.read_csv(out / name)
        merged = ref.merge(new, on=keys, suffixes=("_ref", "_new"), validate="one_to_one")
        assert len(merged) == len(ref)
        for col in ref.select_dtypes("number").columns:
            for _, r in merged.iterrows():
                rows.append({"file": name, **{k: r[k] for k in keys}, "column": col,
                             "reference": r[f"{col}_ref"], "reproduced": r[f"{col}_new"],
                             "absolute_difference": abs(r[f"{col}_new"] - r[f"{col}_ref"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- run
def run(out, limit, csv):
    hist = D.load_us_history(str(csv))
    values, feats = hist.to_numpy(float), list(D.FEATURES)
    origins = [i for i, p in enumerate(hist.index) if p.quarter == 4 and 2004 <= p.year <= 2022][:limit]
    names = [str(hist.index[o]) for o in origins]
    crisis = np.array([RES.crisis_flag(hist.index[o], "US", HORIZON)[0] for o in origins])
    out.mkdir(parents=True, exist_ok=True)
    seed_rows, pit_rows, t0 = [], [], time.time()
    for oi, o in enumerate(origins):
        arr, x0, realised = values[:o + 1], values[o], values[o + 1:o + 1 + HORIZON]
        std = arr.std(0, ddof=1)
        banks = fit_banks(arr, feats, D.PREFERRED, RES.stable_seed(7, "gib_fit", names[oi]))
        pits = {v: [] for v in VARIANTS}
        for seed in SEEDS:
            sample_seed = RES.stable_seed(seed, "unconditional", "gibvar_native", names[oi])
            for v in VARIANTS:                      # same seed -> same assignments and blocks
                paths = G.sample_unconditional(banks[v], N_PATHS, HORIZON,
                                               np.random.default_rng(sample_seed), x0)
                s = SC.score_seed_origin(paths, realised, std, feats)
                seed_rows.append({"origin": names[oi], "seed": seed, "method": v,
                                  **{dest: float(s[src]) for src, dest in METRICS.items()}})
                pits[v].append(s["pits"])
        for v in VARIANTS:                          # average each cell's PIT over seeds
            for cell, pit in enumerate(np.mean(pits[v], axis=0)):
                pit_rows.append({"origin": names[oi], "method": v, "cell": cell, "pit": pit})
        crps = {v: np.mean([r["crps"] for r in seed_rows if r["origin"] == names[oi] and r["method"] == v])
                for v in VARIANTS}
        print(f"[{time.time()-t0:4.0f}s] {names[oi]} ({oi+1}/{len(origins)})  "
              f"soft {crps['full_gibvar']:.4f}  equal {crps['equal_penalty']:.4f}  "
              f"forced {crps['forced_graph']:.4f}", flush=True)

    seed_scores, cell_pits = pd.DataFrame(seed_rows), pd.DataFrame(pit_rows)
    seed_scores.to_csv(out / "scores_by_seed_origin.csv", index=False)
    by_origin, summary = summarise(seed_scores, cell_pits, names, crisis)
    by_origin.to_csv(out / "scores_by_origin.csv", index=False)
    for regime, name in (("overall", "table.csv"), ("crisis", "crisis_table.csv"), ("calm", "calm_table.csv")):
        summary[summary.regime.eq(regime)].to_csv(out / name, index=False)
    diffs = differences(by_origin, names, crisis)
    diffs.drop(columns=["ci95_low", "ci95_high"]).to_csv(out / "matched_differences.csv", index=False)
    diffs.to_csv(out / "paired_intervals.csv", index=False)
    print(summary[summary.regime.eq("overall")][["model", *PROPER]].to_string(index=False))
    if len(origins) == 19:
        check = compare_with_reference(out)
        check.to_csv(out / "reference_comparison.csv", index=False)
        print(f"reference comparison: {len(check)} entries, "
              f"largest absolute difference {check.absolute_difference.max():.2e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=HERE / "runs" / "new_run")
    ap.add_argument("--limit", type=int, default=None, help="first N origins only (smoke test)")
    ap.add_argument("--us", default=None, help="US history CSV (default: data/us/...)")
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        sys.exit(f"output folder is not empty: {a.out}")
    run(a.out, a.limit, a.us or ROOT / "data" / "us" / "2026_Final_Historic_Domestic.csv")
