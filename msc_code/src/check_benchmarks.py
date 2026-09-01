"""Run every model of the report on the rolling-origin tasks and write the results.

For each domain (US: 19 origins, UK: 8) and both tasks it fits the six models
at every origin, generates 1,000 paths under seeds {7, 19, 43}, scores them,
and writes the results CSV set to results/<task>_<domain>/ (summary.csv --
the report's table -- plus crps_by_origin.csv, scores_by_origin.csv,
summary_by_regime.csv and pairwise_differences.csv).  It then prints each
generator's mean CRPS next to the report's value.

Report rows -- US unconditional: GIB-VAR 0.6293, Gaussian VAR 0.7943, BVAR
0.6563, AR-t 0.6496; US conditional: Gaussian VAR 0.7338, AR-t 0.6507,
Gaussian GIB-VAR 0.6002, Student-t GIB-VAR 0.6021, BVAR 0.6184.  UK: 0.8378 /
1.1570 / 0.9026 / 0.8066 and 1.0653 / 0.8741 / 0.9940 / 0.9784 / 0.9615.
Every row reproduces digit-for-digit except Student-t, which lands within a
few thousandths (SMC noise; see student_t.py).  About ten minutes for the US,
four for the UK.

  python src/check_benchmarks.py                       # US and UK, both tasks
  python src/check_benchmarks.py --domain US --limit 3 # quick smoke
  python src/check_benchmarks.py --uk path/to/uk_panel.csv   # UK panel elsewhere
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT / "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)
import data as D                                                    # noqa: E402
import results as RES                                               # noqa: E402
import scoring as SC                                                # noqa: E402
from ar_t import fit_ar_t, sample_ar_t                              # noqa: E402
from bvar import fit_minnesota_bvar                                 # noqa: E402
from gaussian_adapter import complete, sample_gaussian_law          # noqa: E402
from gaussian_var import fit_gaussian_var                           # noqa: E402
from gibvar import fit_gibvar, sample_unconditional                 # noqa: E402
from student_t import clean_bridge_sample                           # noqa: E402

HORIZON, N_PATHS, SEEDS = 12, 1000, (7, 19, 43)
PARTICLES, TAU, NU = 10_000, 0.75, 20.0
UNCON_ROWS = ("GIB-VAR", "Gaussian VAR", "Minnesota BVAR", "AR-t")
COND_ROWS = ("Gaussian VAR", "AR-t", "Gaussian GIB-VAR", "Student-t GIB-VAR", "Minnesota BVAR")
KEY = {"GIB-VAR": "gibvar_native", "Gaussian VAR": "gaussian_var", "Minnesota BVAR": "minnesota_bvar",
       "AR-t": "univariate_ar_t", "Gaussian GIB-VAR": "gibvar_gaussian",
       "Student-t GIB-VAR": "gibvar_student_t"}
THESIS = {("US", "unconditional"): {"GIB-VAR": 0.6293, "Gaussian VAR": 0.7943,
                                    "Minnesota BVAR": 0.6563, "AR-t": 0.6496},
          ("US", "conditional"): {"Gaussian VAR": 0.7338, "AR-t": 0.6507, "Gaussian GIB-VAR": 0.6002,
                                  "Student-t GIB-VAR": 0.6021, "Minnesota BVAR": 0.6184},
          ("UK", "unconditional"): {"GIB-VAR": 0.8378, "Gaussian VAR": 1.1570,
                                    "Minnesota BVAR": 0.9026, "AR-t": 0.8066},
          ("UK", "conditional"): {"Gaussian VAR": 1.0653, "AR-t": 0.8741, "Gaussian GIB-VAR": 0.9940,
                                  "Student-t GIB-VAR": 0.9784, "Minnesota BVAR": 0.9615}}

US = {"name": "US", "features": list(D.FEATURES), "graph": D.PREFERRED, "own_lag_means": None,
      "bvar_max_iter": 20000, "cond_feature": "unemployment", "years": (2004, 2022),
      "load": D.load_us_history, "csv": ROOT / "data" / "us" / "2026_Final_Historic_Domestic.csv"}
UK = {"name": "UK", "features": list(D.UK_FEATURES), "graph": D.UK_PREFERRED,
      "own_lag_means": D.UK_OWN_LAG, "bvar_max_iter": 250000,
      "cond_feature": "unemployment_rate_pct", "years": (2015, 2022), "load": D.load_uk_history,
      "csv": ROOT / "data" / "uk" / "uk_macro_research16_historical.csv"}


def stable_seed(base, *parts):
    digest = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=4).digest()
    return int(base + int.from_bytes(digest, "little") % 100_000)


def q4_origins(hist, domain):
    lo, hi = domain["years"]
    return [i for i, p in enumerate(hist.index) if p.quarter == 4 and lo <= p.year <= hi]


def run(domain, limit, out_root):
    hist = domain["load"](str(domain["csv"]))
    values = hist.to_numpy(float)
    feats = domain["features"]
    cond = domain["cond_feature"]
    ci = feats.index(cond)
    origins = q4_origins(hist, domain)[:limit]
    own = (None if domain["own_lag_means"] is None
           else [domain["own_lag_means"].get(f, 0.9) for f in feats])
    uncon = {k: [] for k in UNCON_ROWS}
    con = {k: [] for k in COND_ROWS}
    t0 = time.time()
    for oi, o in enumerate(origins):
        name = str(hist.index[o])
        arr = values[:o + 1]
        gib = fit_gibvar(arr, feats, domain["graph"], stable_seed(7, "gib_fit", name))
        gv = [fit_gaussian_var(arr)]
        bv = fit_minnesota_bvar(arr, stable_seed(211, "bvar_fit", name), own_lag_means=own,
                                max_iter=domain["bvar_max_iter"])
        ar = fit_ar_t(arr)

        x0, realised = values[o], values[o + 1:o + 1 + HORIZON]
        std = arr.std(0, ddof=1)
        supplied = np.zeros_like(realised)
        supplied[:, ci] = realised[:, ci]
        observed = realised[:, [ci]]
        r7, s7 = np.delete(realised, ci, 1), np.delete(std, ci)
        f7 = [f for f in feats if f != cond]
        per_u = {k: [] for k in UNCON_ROWS}
        per_c = {k: [] for k in COND_ROWS}
        for seed in SEEDS:
            def rng(task, row):
                return np.random.default_rng(stable_seed(seed, task, KEY[row], name))
            u = {"GIB-VAR": sample_unconditional(gib, N_PATHS, HORIZON, rng("unconditional", "GIB-VAR"), x0),
                 "Gaussian VAR": sample_gaussian_law(gv, N_PATHS, HORIZON, rng("unconditional", "Gaussian VAR"), x0),
                 "Minnesota BVAR": sample_gaussian_law(bv, N_PATHS, HORIZON, rng("unconditional", "Minnesota BVAR"), x0),
                 "AR-t": sample_ar_t(ar, N_PATHS, HORIZON, rng("unconditional", "AR-t"), x0)}
            for k in UNCON_ROWS:
                per_u[k].append(SC.score_seed_origin(u[k], realised, std, feats))

            ar_c = sample_ar_t(ar, N_PATHS, HORIZON, rng("conditional", "AR-t"), x0)
            ar_c[:, :, ci] = supplied[None, :, ci]
            t_paths, _ = clean_bridge_sample(gib, observed, [ci], n_paths=N_PATHS,
                                             n_particles=PARTICLES,
                                             rng=rng("conditional", "Student-t GIB-VAR"),
                                             jumpoff=x0, block_length=4, tau=TAU, nu=NU,
                                             ess_ratio=0.5)
            c = {"Gaussian VAR": complete(gv, supplied, [ci], N_PATHS, rng("conditional", "Gaussian VAR"), x0)[0],
                 "AR-t": ar_c,
                 "Gaussian GIB-VAR": complete(gib, supplied, [ci], N_PATHS, rng("conditional", "Gaussian GIB-VAR"), x0)[0],
                 "Student-t GIB-VAR": t_paths,
                 "Minnesota BVAR": complete(bv, supplied, [ci], N_PATHS, rng("conditional", "Minnesota BVAR"), x0)[0]}
            for k in COND_ROWS:
                assert np.abs(c[k][:, :, ci] - realised[:, ci]).max() < 1e-8
                per_c[k].append(SC.score_seed_origin(np.delete(c[k], ci, 2), r7, s7, f7))
        for k in UNCON_ROWS:
            uncon[k].append(per_u[k])
        for k in COND_ROWS:
            con[k].append(per_c[k])
        print(f"[{time.time()-t0:4.0f}s] {domain['name']} {name} ({oi+1}/{len(origins)})  "
              f"uncond GIB-VAR {np.mean([s['crps'] for s in per_u['GIB-VAR']]):.4f}  "
              f"cond Student-t {np.mean([s['crps'] for s in per_c['Student-t GIB-VAR']]):.4f}",
              flush=True)

    names = [str(hist.index[o]) for o in origins]
    for task, table in (("unconditional", uncon), ("conditional", con)):
        target = RES.write(task, domain["name"], names, table, SEEDS,
                           Path(out_root) / f"{task}_{domain['name']}")
        print(f"\n{domain['name']} {task}: written to {target}")
        print(f"  {'generator':20s} {'CRPS':>8s} {'report':>8s}")
        for k in table:
            print(f"  {k:20s} {SC.aggregate(table[k])['crps']:8.4f} {THESIS[domain['name'], task][k]:8.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--domain", choices=("US", "UK", "both"), default="both")
    ap.add_argument("--limit", type=int, default=None, help="first N origins only (smoke test)")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--us", default=None, help="US history CSV (default: data/us/...)")
    ap.add_argument("--uk", default=None, help="UK panel CSV (default: data/uk/...)")
    a = ap.parse_args()
    if a.us:
        US["csv"] = Path(a.us)
    if a.uk:
        UK["csv"] = Path(a.uk)
    if a.domain in ("US", "both"):
        run(US, a.limit, a.out)
    if a.domain in ("UK", "both"):
        run(UK, a.limit, a.out)
