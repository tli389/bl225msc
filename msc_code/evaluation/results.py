"""Write rolling scores, regime summaries and paired bootstrap intervals."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

BLOCK_LENGTH, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED = 3, 2000, 7
# Crisis-overlap windows and minimum overlap.
CRISIS_WINDOWS = {"US": (("2007Q1", "2010Q4"), ("2019Q3", "2021Q4")),
                  "UK": (("2020Q1", "2021Q4"),)}
MIN_CRISIS_QUARTERS = 4

# Method keys used to derive reproducible seeds.
METHOD_KEY = {"GIB-VAR": "gibvar_native", "Gaussian VAR": "gaussian_var",
              "Minnesota BVAR": "minnesota_bvar", "AR-t": "univariate_ar_t",
              "Gaussian GIB-VAR": "gibvar_gaussian",
              "Student-t GIB-VAR": "gibvar_student_t"}
METRIC_KEY = {"crps": "crps", "qwcrps": "qwcrps", "energy": "energy_score",
              "variogram": "variogram_score"}
PAIRS = {"unconditional": [("GIB-VAR", "Minnesota BVAR"),
                           ("Gaussian VAR", "Minnesota BVAR"),
                           ("GIB-VAR", "AR-t")],
         "conditional": [("Gaussian GIB-VAR", "Minnesota BVAR"),
                         ("Student-t GIB-VAR", "Minnesota BVAR"),
                         ("Student-t GIB-VAR", "Gaussian GIB-VAR"),
                         ("Student-t GIB-VAR", "AR-t")]}
SCORE_KEYS = ("crps", "qwcrps", "energy", "variogram", "cov80", "cov95",
              "width80", "width95")


def stable_seed(base_seed, *parts):
    key = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return int(base_seed + int.from_bytes(digest, "little") % 100_000)


def crisis_flag(origin: pd.Period, domain: str = "US", horizon: int = 12):
    """(crisis-overlap?, overlapping quarters) for the origin's target window."""
    target = pd.period_range(origin + 1, origin + horizon, freq="Q")
    covered = set()
    for begin, end in CRISIS_WINDOWS.get(domain, CRISIS_WINDOWS["US"]):
        covered.update(target.intersection(pd.period_range(begin, end, freq="Q")))
    return len(covered) >= MIN_CRISIS_QUARTERS, len(covered)


def moving_block_resample(n, block_length, rng):
    """Resample circular blocks of origin positions."""
    length = min(max(int(block_length), 1), n)
    n_blocks = int(np.ceil(n / length))
    starts = rng.integers(0, n, size=n_blocks)
    positions = np.concatenate([(s + np.arange(length)) % n for s in starts])
    return positions[:n]


def block_bootstrap_interval(differences, crisis, regime, seed):
    """2,000 resampled means of the per-origin differences -> 2.5/97.5 pct."""
    rng = np.random.default_rng(seed)
    draws, attempts = [], 0
    while len(draws) < BOOTSTRAP_DRAWS:
        attempts += 1
        if attempts > max(100, 20 * BOOTSTRAP_DRAWS):
            raise RuntimeError(f"unable to bootstrap non-empty {regime} sample")
        chosen = moving_block_resample(len(differences), BLOCK_LENGTH, rng)
        if regime != "overall":
            chosen = chosen[crisis[chosen] == (regime == "crisis")]
            if not len(chosen):
                continue
        draws.append(float(np.mean(differences[chosen])))
    draws = np.asarray(draws)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def write(task, domain, origin_names, per_origin, seeds, out_dir):
    """per_origin: {generator: [[score dict per seed] per origin]}."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    generators = list(per_origin)
    origins = [pd.Period(name, freq="Q") for name in origin_names]
    flags = [crisis_flag(o, domain) for o in origins]
    crisis = np.array([f[0] for f in flags])

    # Store every seed-origin score.
    rows = []
    for gen in generators:
        for i, name in enumerate(origin_names):
            for r, seed in enumerate(seeds):
                sc = per_origin[gen][i][r]
                rows.append({"task": task, "domain": domain, "origin": name,
                             "crisis_overlap": bool(crisis[i]),
                             "crisis_quarters": flags[i][1], "seed": seed,
                             "generator": gen,
                             **{k: float(sc[k]) for k in SCORE_KEYS}})
    long = pd.DataFrame(rows)
    long.to_csv(out / "scores_by_origin.csv", index=False, float_format="%.6f")

    # Average seeds within each origin.
    by_origin = (long.groupby(["origin", "generator"], sort=False)[list(SCORE_KEYS)]
                 .mean().reset_index())
    wide = by_origin.pivot(index="origin", columns="generator", values="crps")
    wide = wide.loc[origin_names, generators]
    wide.insert(0, "crisis_overlap", crisis)
    wide["best"] = wide[generators].idxmin(axis=1)
    wide.to_csv(out / "crps_by_origin.csv", float_format="%.6f")

    # Average origins and pool the seed-averaged PITs.
    from scipy import stats
    summary = []
    for gen in generators:
        row = {"generator": gen}
        for k in SCORE_KEYS:
            row[k] = float(np.mean([np.mean([s[k] for s in seeds_])
                                    for seeds_ in per_origin[gen]]))
        pooled = np.concatenate([np.mean([s["pits"] for s in seeds_], axis=0)
                                 for seeds_ in per_origin[gen]])
        row["pit_ks"] = float(stats.kstest(pooled, "uniform").statistic)
        row["n_origins"] = len(origin_names)
        summary.append(row)
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False,
                                 float_format="%.6f")

    # Split summaries into crisis and calm origins.
    regime_rows = []
    for gen in generators:
        g = by_origin[by_origin.generator == gen].set_index("origin").loc[origin_names]
        for regime in ("overall", "crisis", "calm"):
            mask = (np.ones(len(origin_names), bool) if regime == "overall"
                    else crisis == (regime == "crisis"))
            if not mask.any():
                continue
            regime_rows.append({"generator": gen, "regime": regime,
                                "n_origins": int(mask.sum()),
                                **{k: float(g[k].to_numpy()[mask].mean())
                                   for k in SCORE_KEYS}})
    pd.DataFrame(regime_rows).to_csv(out / "summary_by_regime.csv", index=False,
                                     float_format="%.6f")

    # Paired differences and circular block-bootstrap intervals.
    rows = []
    for a, b in PAIRS[task]:
        if a not in per_origin or b not in per_origin:
            continue
        for metric in ("crps", "qwcrps", "energy", "variogram"):
            va = by_origin[by_origin.generator == a].set_index("origin").loc[origin_names, metric].to_numpy()
            vb = by_origin[by_origin.generator == b].set_index("origin").loc[origin_names, metric].to_numpy()
            diff = va - vb
            for regime in ("overall", "calm", "crisis"):
                mask = (np.ones(len(diff), bool) if regime == "overall"
                        else crisis == (regime == "crisis"))
                if not mask.any():
                    continue
                seed = stable_seed(BOOTSTRAP_SEED, "paired", "raw", task,
                                   METHOD_KEY[a], METHOD_KEY[b],
                                   METRIC_KEY[metric], regime)
                lo, hi = block_bootstrap_interval(diff, crisis, regime, seed)
                rows.append({"task": task, "domain": domain, "metric": metric,
                             "model_a": a, "model_b": b, "regime": regime,
                             "n_origins": int(mask.sum()),
                             "a_wins": int((diff[mask] < 0).sum()),
                             "mean_difference_a_minus_b": float(diff[mask].mean()),
                             "ci95_low": lo, "ci95_high": hi,
                             "interval_includes_zero": bool(lo <= 0.0 <= hi),
                             "bootstrap": "circular moving blocks over origins",
                             "block_length": BLOCK_LENGTH,
                             "bootstrap_draws": BOOTSTRAP_DRAWS,
                             "seed": seed})
    pd.DataFrame(rows).to_csv(out / "pairwise_differences.csv", index=False,
                              float_format="%.6f")

    (out / "README.md").write_text(_README.format(task=task, domain=domain,
                                                  n=len(origin_names)))
    return out


_README = """
"""
