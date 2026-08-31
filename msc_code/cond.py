"""Conditional completion evaluation, US or UK -- all five thesis rows.

The realised unemployment path is supplied over each 12-quarter window; the
other seven variables are completed and scored.  Model files: gibvar.py +
gaussian.py + student_t.py for the two GIB-VAR completions, gaussian_var.py,
bvar.py and ar_t.py for the benchmarks.  Locked protocols: seeds {7, 19,
43}, 1000 paths, 10,000 bridge particles, tau 0.75, nu 20, CESS 0.5.

Expected CRPS -- US: Gaussian VAR 0.7338, AR-t 0.6507, Gaussian GIB 0.6002,
Student-t GIB 0.6021, BVAR 0.6184.  UK (thesis-reported Q4 subset):
Gaussian VAR 1.0653, AR-t 0.8741, Gaussian GIB 0.9940, Student-t GIB
0.9784, BVAR 0.9615.  Student-t rows move ~0.005 across machines (SMC
resampling amplifies floating-point noise).

  python cond.py                       # US, data/us/2026_Final_Historic_Domestic.csv
  python cond.py --uk                  # UK, data/uk/uk_macro_research16_historical.csv
  python cond.py --uk path/to/uk_history.csv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import results as RES                                   # noqa: E402
import scoring as SC                                    # noqa: E402
from gaussian import conditional_mixture_sample         # noqa: E402
from student_t import block_bridge_conditional_sample   # noqa: E402
from uncon import (DEFAULT_UK_CSV, DEFAULT_US_CSV, HORIZON,  # noqa: E402
                   N_PATHS, SEEDS, UK, US, fit_benchmarks, fit_origin,
                   q4_origins, stable_seed)

PARTICLES, TAU, NU = 10_000, 0.75, 20.0
ROWS = ("Gaussian VAR", "AR-t", "Gaussian GIB-VAR", "Student-t GIB-VAR",
        "Minnesota BVAR")
THESIS = {("US", "Gaussian VAR"): 0.7338, ("US", "AR-t"): 0.6507,
          ("US", "Gaussian GIB-VAR"): 0.6002,
          ("US", "Student-t GIB-VAR"): 0.6021,
          ("US", "Minnesota BVAR"): 0.6184,
          ("UK", "Gaussian VAR"): 1.0653, ("UK", "AR-t"): 0.8741,
          ("UK", "Gaussian GIB-VAR"): 0.9940,
          ("UK", "Student-t GIB-VAR"): 0.9784,
          ("UK", "Minnesota BVAR"): 0.9615}


def main(csv_path: str, domain=US, out_dir: str = "results") -> None:
    feats = domain["features"]
    cond = domain["cond_feature"]
    ci = feats.index(cond) if isinstance(feats, list) else \
        list(feats).index(cond)
    hist = domain["load"](csv_path)
    values = hist.to_numpy(float)
    origins = q4_origins(hist, domain)
    results = {k: [] for k in ROWS}
    t0 = time.time()
    for oi, o in enumerate(origins):
        origin_name = str(hist.index[o])
        gib = fit_origin(hist, o, domain)
        gv, bv, ar = fit_benchmarks(hist, o, domain)
        x0, realised = values[o], values[o + 1:o + 1 + HORIZON]
        std = values[:o + 1].std(0, ddof=1)
        supplied = np.zeros_like(realised)
        supplied[:, ci] = realised[:, ci]       # full-width supplied path
        observed = realised[:, [ci]]            # bridge-style observed column
        r7, s7 = np.delete(realised, ci, 1), np.delete(std, ci)
        f7 = [f for f in feats if f != cond]
        per = {k: [] for k in ROWS}
        for seed in SEEDS:
            def rng_for(method):
                return np.random.default_rng(stable_seed(
                    seed, "conditional", method, origin_name))
            gvp, _ = conditional_mixture_sample(
                gv, supplied, [cond], N_PATHS, rng_for("gaussian_var"), x0,
                features=feats, bounds=domain["bounds"],
                apply_hidden_bounds=False)
            arp, _ = ar.sample_conditional(
                supplied, [cond], N_PATHS, rng_for("univariate_ar_t"),
                jumpoff=x0, apply_hidden_bounds=False)
            g, _ = conditional_mixture_sample(
                gib, supplied, [cond], N_PATHS, rng_for("gibvar_gaussian"),
                x0, features=feats, bounds=domain["bounds"],
                apply_hidden_bounds=False)
            t, _ = block_bridge_conditional_sample(
                gib, observed, [cond], n_paths=N_PATHS,
                n_particles=PARTICLES, rng=rng_for("gibvar_student_t"),
                jumpoff=x0, block_length=4, tau=TAU, kernel="student_t",
                degrees_of_freedom=NU, ess_resample_ratio=0.5,
                likelihood_tempering=True, tempering_cess_ratio=0.5,
                max_tempering_steps=50, rejuvenate_after_resampling=True,
                apply_hidden_bounds=False, features=feats,
                bounds=domain["bounds"])
            bvp, _ = conditional_mixture_sample(
                bv, supplied, [cond], N_PATHS, rng_for("minnesota_bvar"),
                x0, features=feats, bounds=domain["bounds"],
                apply_hidden_bounds=False)
            runs = {"Gaussian VAR": gvp, "AR-t": arp,
                    "Gaussian GIB-VAR": g, "Student-t GIB-VAR": t,
                    "Minnesota BVAR": bvp}
            for name, paths in runs.items():
                assert np.abs(paths[:, :, ci] - realised[:, ci]).max() < 1e-8
                per[name].append(SC.score_seed_origin(
                    np.delete(paths, ci, 2), r7, s7, f7))
        for k in ROWS:
            results[k].append(per[k])
        print(f"[{time.time()-t0:5.0f}s] {origin_name}: all five "
              f"completions done ({oi+1}/{len(origins)})", flush=True)
    subset = " (thesis-reported Q4 subset)" if domain is UK else ""
    print(f"\n{domain['name']} conditional, {len(origins)} origins x "
          f"3 seeds{subset}:")
    print(f"{'Generator':18s} {'CRPS':>7s} {'thesis':>7s} {'qWCRPS':>7s} "
          f"{'Energy':>8s} {'Vario':>7s} {'Cov80':>6s} {'Cov95':>6s}")
    for k in ROWS:
        a = SC.aggregate(results[k])
        print(f"{k:18s} {a['crps']:7.4f} {THESIS[domain['name'], k]:7.4f} "
              f"{a['qwcrps']:7.4f} {a['energy']:8.4f} "
              f"{a['variogram']:7.4f} {a['cov80']:6.1%} {a['cov95']:6.1%}")

    target = RES.write("conditional", domain["name"],
                       [str(hist.index[o]) for o in origins], results, SEEDS,
                       Path(out_dir) / f"conditional_{domain['name']}")
    print(f"\nresults written to {target}  (summary.csv, summary_by_regime.csv, "
          "crps_by_origin.csv, scores_by_origin.csv, pairwise_differences.csv)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="?",
                        help="US Fed historic CSV (default: data/us/"
                             "2026_Final_Historic_Domestic.csv)")
    parser.add_argument("--uk", nargs="?", const=str(DEFAULT_UK_CSV),
                        metavar="UK_CSV",
                        help="run the UK evaluation on the licensed panel "
                             "(default location: data/uk/"
                             "uk_macro_research16_historical.csv)")
    parser.add_argument("--out", default="results",
                        help="folder for the CSV results (default: results/)")
    args = parser.parse_args()
    if args.uk:
        main(args.uk, UK, args.out)
    else:
        main(args.csv or str(DEFAULT_US_CSV), US, args.out)
