"""Unconditional rolling-origin evaluation, US or UK -- four generators.

Model files in this folder: gibvar.py, gaussian_var.py, bvar.py, ar_t.py.
US: 19 Q4 origins 2004--2022 (thesis Panel A).  UK: the licensed panel via
--uk; the 8 Q4 origins 2015--2022 that the thesis reports (its Q4 subset
of the 29-origin grid).  Locked protocols throughout: seeds {7, 19, 43},
25 GIB systems, 100 BVAR posterior draws, 1000 paths, engine seed scheme.

Expected CRPS -- US: GIB-VAR 0.6293, Gaussian VAR 0.7943, BVAR 0.6563,
AR-t 0.6496.  UK: GIB-VAR 0.8378, Gaussian VAR 1.1570, BVAR 0.9026,
AR-t 0.8066.

  python uncon.py                      # US, data/us/2026_Final_Historic_Domestic.csv
  python uncon.py --uk                 # UK, data/uk/uk_macro_research16_historical.csv
  python uncon.py --uk path/to/uk_history.csv
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
DATA = HERE / "data"                       # all input data lives here
DEFAULT_US_CSV = DATA / "us" / "2026_Final_Historic_Domestic.csv"
DEFAULT_UK_CSV = DATA / "uk" / "uk_macro_research16_historical.csv"
import results as RES                                   # noqa: E402
import scoring as SC                                    # noqa: E402
from ar_t import UnivariateARGenerator                  # noqa: E402
from bvar import MinnesotaPosteriorVARGenerator         # noqa: E402
from data import (FEATURES, PREFERRED, UK_BOUNDS, UK_FEATURES,  # noqa: E402
                  UK_OWN_LAG, UK_PREFERRED, load_uk_history,
                  load_us_history)
from gaussian import gaussian_mixture_sample            # noqa: E402
from gaussian_var import LinearPathGenerator            # noqa: E402
from gibvar import GIBVAR                               # noqa: E402

HORIZON, N_PATHS, B_REFITS, SEEDS = 12, 1000, 25, (7, 19, 43)

# locked us_class8 sanity bounds (protocol values, this folder's column names)
BOUNDS_US = {"gdp_growth": (-40.0, 40.0), "unemployment": (0.0, 35.0),
             "treasury_3m": (0.0, 20.0), "treasury_10y": (0.0, 25.0),
             "bbb_spread": (0.0, 15.0), "hpi_growth": (-25.0, 20.0),
             "cre_growth": (-40.0, 30.0), "equity_growth": (-80.0, 80.0)}

US = {"name": "US", "features": FEATURES, "graph": PREFERRED,
      "bounds": BOUNDS_US, "own_lag_means": None, "bvar_max_iter": 20000,
      "cond_feature": "unemployment", "years": (2004, 2022), "n_origins": 19,
      "load": load_us_history}
UK = {"name": "UK", "features": UK_FEATURES, "graph": UK_PREFERRED,
      "bounds": UK_BOUNDS, "own_lag_means": UK_OWN_LAG,
      "bvar_max_iter": 250000, "cond_feature": "unemployment_rate_pct",
      "years": (2015, 2022), "n_origins": 8, "load": load_uk_history}

ROWS = ("GIB-VAR", "Gaussian VAR", "Minnesota BVAR", "AR-t")
THESIS = {("US", "GIB-VAR"): 0.6293, ("US", "Gaussian VAR"): 0.7943,
          ("US", "Minnesota BVAR"): 0.6563, ("US", "AR-t"): 0.6496,
          ("UK", "GIB-VAR"): 0.8378, ("UK", "Gaussian VAR"): 1.1570,
          ("UK", "Minnesota BVAR"): 0.9026, ("UK", "AR-t"): 0.8066}


def stable_seed(base_seed, *parts):
    """Verbatim from the package's envelope.stable_seed."""
    key = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=4).digest()
    return int(base_seed + int.from_bytes(digest, "little") % 100_000)


def q4_origins(hist, domain):
    lo, hi = domain["years"]
    origins = [i for i, p in enumerate(hist.index)
               if p.quarter == 4 and lo <= p.year <= hi]
    assert len(origins) == domain["n_origins"], "unexpected origin count"
    return origins


def fit_origin(hist, o, domain=US):
    """The locked-protocol GIB-VAR fit; also used by cond.py."""
    origin_name = str(hist.index[o])
    return GIBVAR(n_dynamics=B_REFITS, block_length=4, expert_weight=0.35,
                  max_parents=3, max_spectral_radius=0.995,
                  random_state=stable_seed(7, "gib_fit", origin_name),
                  path_solver="lars", features=domain["features"],
                  expert_edges=domain["graph"],
                  bounds=None).fit(hist.iloc[:o + 1])


def fit_benchmarks(hist, o, domain=US):
    """Locked-protocol fits of the three benchmarks; also used by cond.py."""
    origin_name = str(hist.index[o])
    train = hist.iloc[:o + 1]
    gv = LinearPathGenerator("dense", "gaussian", features=domain["features"],
                             bounds=domain["bounds"]).fit(train)
    ar = UnivariateARGenerator(features=domain["features"], bounds=None,
                               max_spectral_radius=0.995,
                               minimum_degrees_of_freedom=2.1,
                               maximum_degrees_of_freedom=30.0).fit(train)
    bv = MinnesotaPosteriorVARGenerator(
        n_dynamics=100, burn_in=150, thin=3,
        max_sampling_iterations=domain["bvar_max_iter"],
        seed=stable_seed(211, "bvar_fit", origin_name),
        features=domain["features"], bounds=domain["bounds"],
        own_lag_means=domain["own_lag_means"]).fit(train)
    return gv, bv, ar


def main(csv_path: str, domain=US, out_dir: str = "results") -> None:
    feats = domain["features"]
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
        per = {k: [] for k in ROWS}
        for seed in SEEDS:
            def rng_for(method):
                return np.random.default_rng(stable_seed(
                    seed, "unconditional", method, origin_name))
            runs = {
                "GIB-VAR": gib.sample(N_PATHS, HORIZON,
                                      rng_for("gibvar_native"), jumpoff=x0),
                "Gaussian VAR": gaussian_mixture_sample(
                    gv, N_PATHS, HORIZON, rng_for("gaussian_var"), x0,
                    features=feats, bounds=domain["bounds"],
                    apply_bounds=False),
                "Minnesota BVAR": gaussian_mixture_sample(
                    bv, N_PATHS, HORIZON, rng_for("minnesota_bvar"), x0,
                    features=feats, bounds=domain["bounds"],
                    apply_bounds=False),
                "AR-t": ar.sample(N_PATHS, HORIZON,
                                  rng_for("univariate_ar_t"), jumpoff=x0,
                                  apply_bounds=False),
            }
            for k in ROWS:
                per[k].append(SC.score_seed_origin(runs[k], realised, std,
                                                   feats))
        for k in ROWS:
            results[k].append(per[k])
        print(f"[{time.time()-t0:5.0f}s] {origin_name}: "
              f"{len(gib.draws_)} GIB systems, "
              f"{len(bv.draws_)} BVAR draws ({oi+1}/{len(origins)})",
              flush=True)
    subset = " (thesis-reported Q4 subset)" if domain is UK else ""
    print(f"\n{domain['name']} unconditional, {len(origins)} origins x "
          f"3 seeds{subset}:")
    print(f"{'Generator':15s} {'CRPS':>7s} {'thesis':>7s} {'qWCRPS':>7s} "
          f"{'Energy':>8s} {'Vario':>7s} {'Cov80':>6s} {'Cov95':>6s} "
          f"{'PIT-KS':>7s}")
    for k in ROWS:
        a = SC.aggregate(results[k])
        print(f"{k:15s} {a['crps']:7.4f} {THESIS[domain['name'], k]:7.4f} "
              f"{a['qwcrps']:7.4f} {a['energy']:8.4f} "
              f"{a['variogram']:7.4f} {a['cov80']:6.1%} {a['cov95']:6.1%} "
              f"{a['pit_ks']:7.4f}")

    target = RES.write("unconditional", domain["name"],
                       [str(hist.index[o]) for o in origins], results, SEEDS,
                       Path(out_dir) / f"unconditional_{domain['name']}")
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
