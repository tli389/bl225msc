"""Fed 2024 Exploratory Conditions B narrative completion -- Student-t vs BVAR.

The three reported masks each supply two designed paths from the 2023Q4
jump-off; the remaining six variables are completed and scored against the
official reference path.  Fit once per jump-off (shared GIB bank; Minnesota
BVAR), locked narrative seed scheme, common random numbers across masks.

Expected CRPS (sealed pipeline on this machine / thesis print):
  activity-labour       Student-t 1.1460          BVAR 0.9406 (best, = thesis)
  property              Student-t 0.7082 (best; thesis 0.7057)   BVAR 0.8825
  high-rate recession   Student-t 1.1739 (best; thesis 1.1698)   BVAR 1.1792
Student-t rows carry the ~0.005 cross-machine SMC wobble.

  python narrative.py                  # data/us/2024-Table_2A_Historic_Domestic.csv
                                       # + data/us/2024-Table_Exploratory_Macro_Conditions_B_Domestic.csv
  python narrative.py historic_csv conditions_b_csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scoring as SC                                    # noqa: E402
from bvar import MinnesotaPosteriorVARGenerator         # noqa: E402
from data import FEATURES, PREFERRED, load_conditions_b  # noqa: E402
from gaussian import conditional_mixture_sample         # noqa: E402
from gibvar import GIBVAR                               # noqa: E402
from student_t import block_bridge_conditional_sample   # noqa: E402
from uncon import BOUNDS_US, HORIZON, N_PATHS, SEEDS, stable_seed  # noqa: E402

PARTICLES, TAU, NU = 10_000, 0.75, 20.0
DOMAIN, JUMPOFF = "us_class8", "2023Q4"
MASKS = {"activity-labour": ["gdp_growth", "unemployment"],
         "property": ["hpi_growth", "cre_growth"],
         "high-rate recession": ["unemployment", "treasury_3m"]}
EXPECTED = {("activity-labour", "Student-t GIB-VAR"): 1.1460,
            ("activity-labour", "Minnesota BVAR"): 0.9406,
            ("property", "Student-t GIB-VAR"): 0.7082,
            ("property", "Minnesota BVAR"): 0.8825,
            ("high-rate recession", "Student-t GIB-VAR"): 1.1739,
            ("high-rate recession", "Minnesota BVAR"): 1.1792}


def fit_narrative(hist):
    """One shared fit per jump-off, narrative seed scheme."""
    gib = GIBVAR(n_dynamics=25, block_length=4, expert_weight=0.35,
                 max_parents=3, max_spectral_radius=0.995,
                 random_state=stable_seed(7, "narrative_gib_fit",
                                          DOMAIN, JUMPOFF),
                 path_solver="lars", features=FEATURES,
                 expert_edges=PREFERRED, bounds=None).fit(hist)
    bv = MinnesotaPosteriorVARGenerator(
        n_dynamics=100, burn_in=150, thin=3, max_sampling_iterations=20000,
        seed=stable_seed(211, "narrative_bvar_fit", DOMAIN, JUMPOFF),
        features=FEATURES, bounds=BOUNDS_US, own_lag_means=None).fit(hist)
    return gib, bv


def main(hist_csv: str, cond_csv: str) -> None:
    hist, target = load_conditions_b(hist_csv, cond_csv)
    values = hist.to_numpy(float)
    x0, std = values[-1], values.std(0, ddof=1)
    t0 = time.time()
    gib, bv = fit_narrative(hist)
    print(f"[{time.time()-t0:4.0f}s] fitted once to "
          f"{hist.index[0]}-{hist.index[-1]}: {len(gib.draws_)} GIB "
          f"systems, {len(bv.draws_)} BVAR draws", flush=True)
    for mask, feats in MASKS.items():
        idx = [FEATURES.index(f) for f in feats]
        observed = target[:, idx]
        supplied = np.zeros((HORIZON, len(FEATURES)))
        supplied[:, idx] = observed
        keep = [j for j in range(len(FEATURES)) if j not in idx]
        f_hidden = [FEATURES[j] for j in keep]
        rows: dict[str, list] = {}
        for seed in SEEDS:
            t, _ = block_bridge_conditional_sample(
                gib, observed, feats, n_paths=N_PATHS,
                n_particles=PARTICLES,
                rng=np.random.default_rng(stable_seed(
                    seed, "narrative_completion", DOMAIN,
                    "gibvar_student_t", JUMPOFF)),
                jumpoff=x0, block_length=4, tau=TAU, kernel="student_t",
                degrees_of_freedom=NU, ess_resample_ratio=0.5,
                likelihood_tempering=True, tempering_cess_ratio=0.5,
                max_tempering_steps=50, rejuvenate_after_resampling=True,
                apply_hidden_bounds=False, features=FEATURES,
                bounds=BOUNDS_US)
            b, _ = conditional_mixture_sample(
                bv, supplied, feats, N_PATHS,
                np.random.default_rng(stable_seed(
                    seed, "narrative_completion", DOMAIN,
                    "minnesota_bvar", JUMPOFF)),
                x0, features=FEATURES, bounds=BOUNDS_US,
                apply_hidden_bounds=False)
            for name, paths in (("Student-t GIB-VAR", t),
                                ("Minnesota BVAR", b)):
                assert np.abs(paths[:, :, idx] - observed[None]).max() < 1e-8
                rows.setdefault(name, []).append(SC.score_seed_origin(
                    paths[:, :, keep], target[:, keep], std[keep], f_hidden))
        print(f"\n[{time.time()-t0:4.0f}s] {mask} (supplies {feats}):")
        for name, ss in rows.items():
            m = {k: float(np.mean([s[k] for s in ss]))
                 for k in ("crps", "qwcrps", "energy", "variogram")}
            print(f"  {name:18s} CRPS {m['crps']:6.4f} "
                  f"(expect {EXPECTED[mask, name]:6.4f})  "
                  f"qWCRPS {m['qwcrps']:6.4f}  Energy {m['energy']:8.4f}  "
                  f"Vario {m['variogram']:6.4f}")
    print("\nthesis narrative: activity-labour BVAR 0.9406 best; "
          "property Student-t 0.7057 best; high-rate Student-t 1.1698 best")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        main(sys.argv[1], sys.argv[2])
    else:
        US_DATA = HERE / "data" / "us"
        main(str(US_DATA / "2024-Table_2A_Historic_Domestic.csv"),
             str(US_DATA / "2024-Table_Exploratory_Macro_Conditions_B_Domestic.csv"))
