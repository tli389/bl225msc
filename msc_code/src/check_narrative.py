"""Fed 2024 Exploratory Conditions B narrative completion on the rebuilds.

Same design as the report's narrative study: fit once to the history ending
2023Q4 (shared GIB-VAR bank, Minnesota BVAR); for each of the three masks
supply the two designed paths, complete the other six variables with 1,000
paths under seeds {7, 19, 43}, and score the completed variables against the
official reference path.  Seeds follow the locked narrative scheme.

Report rows (CRPS):
  activity-labour       Gaussian GIB 1.1391  Student-t 1.1435  BVAR 0.9406 (best)
  property              Gaussian GIB 0.7220  Student-t 0.7057 (best)  BVAR 0.8825
  high-rate recession   Gaussian GIB 1.1774  Student-t 1.1698 (best)  BVAR 1.1792
Gaussian GIB-VAR and BVAR reproduce exactly (all four scores); Student-t lands
within its SMC noise.  About a minute.

  python src/check_narrative.py
  python src/check_narrative.py historic_csv conditions_b_csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT / "evaluation")):
    if p not in sys.path:
        sys.path.insert(0, p)
import scoring as SC                                                # noqa: E402
from bvar import fit_minnesota_bvar                                 # noqa: E402
from check_benchmarks import HORIZON, N_PATHS, NU, PARTICLES, SEEDS, TAU, stable_seed  # noqa: E402
from data import FEATURES, PREFERRED, load_conditions_b            # noqa: E402
from gaussian_adapter import complete                               # noqa: E402
from gibvar import fit_gibvar                                       # noqa: E402
from student_t import clean_bridge_sample                           # noqa: E402

DOMAIN, JUMPOFF = "us_class8", "2023Q4"
MASKS = {"activity-labour": ["gdp_growth", "unemployment"],
         "property": ["hpi_growth", "cre_growth"],
         "high-rate recession": ["unemployment", "treasury_3m"]}
THESIS = {("activity-labour", "Gaussian GIB-VAR"): 1.1391,
          ("activity-labour", "Student-t GIB-VAR"): 1.1435,
          ("activity-labour", "Minnesota BVAR"): 0.9406,
          ("property", "Gaussian GIB-VAR"): 0.7220,
          ("property", "Student-t GIB-VAR"): 0.7057,
          ("property", "Minnesota BVAR"): 0.8825,
          ("high-rate recession", "Gaussian GIB-VAR"): 1.1774,
          ("high-rate recession", "Student-t GIB-VAR"): 1.1698,
          ("high-rate recession", "Minnesota BVAR"): 1.1792}


def main(hist_csv, cond_csv):
    hist, target = load_conditions_b(hist_csv, cond_csv)
    values = hist.to_numpy(float)
    feats = list(FEATURES)
    x0, std = values[-1], values.std(0, ddof=1)
    t0 = time.time()
    gib = fit_gibvar(values, feats, PREFERRED, stable_seed(7, "narrative_gib_fit", DOMAIN, JUMPOFF))
    bv = fit_minnesota_bvar(values, stable_seed(211, "narrative_bvar_fit", DOMAIN, JUMPOFF))
    print(f"[{time.time()-t0:4.0f}s] fitted once to {hist.index[0]}-{hist.index[-1]}", flush=True)
    for mask, supplied_feats in MASKS.items():
        idx = [feats.index(f) for f in supplied_feats]
        keep = [j for j in range(len(feats)) if j not in idx]
        observed = target[:, idx]
        supplied = np.zeros((HORIZON, len(feats)))
        supplied[:, idx] = observed
        rows = {"Gaussian GIB-VAR": [], "Student-t GIB-VAR": [], "Minnesota BVAR": []}
        for seed in SEEDS:
            def rng(key):
                return np.random.default_rng(stable_seed(seed, "narrative_completion", DOMAIN, key, JUMPOFF))
            g, _ = complete(gib, supplied, idx, N_PATHS, rng("gibvar_gaussian"), x0)
            t, _ = clean_bridge_sample(gib, observed, idx, n_paths=N_PATHS, n_particles=PARTICLES,
                                       rng=rng("gibvar_student_t"), jumpoff=x0, block_length=4,
                                       tau=TAU, nu=NU, ess_ratio=0.5)
            b, _ = complete(bv, supplied, idx, N_PATHS, rng("minnesota_bvar"), x0)
            for name, paths in (("Gaussian GIB-VAR", g), ("Student-t GIB-VAR", t), ("Minnesota BVAR", b)):
                assert np.abs(paths[:, :, idx] - observed[None]).max() < 1e-8
                rows[name].append(SC.score_seed_origin(paths[:, :, keep], target[:, keep],
                                                       std[keep], [feats[j] for j in keep]))
        print(f"\n[{time.time()-t0:4.0f}s] {mask} (supplies {supplied_feats}):")
        for name, ss in rows.items():
            m = {k: float(np.mean([s[k] for s in ss])) for k in ("crps", "qwcrps", "energy", "variogram")}
            print(f"  {name:18s} CRPS {m['crps']:6.4f} (report {THESIS[mask, name]:6.4f})  "
                  f"qWCRPS {m['qwcrps']:6.4f}  Energy {m['energy']:8.4f}  Vario {m['variogram']:6.4f}")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        main(str(ROOT / "data" / "us" / "2024-Table_2A_Historic_Domestic.csv"),
             str(ROOT / "data" / "us" / "2024-Table_Exploratory_Macro_Conditions_B_Domestic.csv"))
