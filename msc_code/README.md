# msc_code -- the report's methods, one file each

Everything runs from this folder.  Requirements: numpy, pandas,
scikit-learn, scipy (no package installation needed).

## Model files (one method of the report per file)

| file | report method |
|---|---|
| `gibvar.py` | GIB-VAR: graph-weighted sparse estimation (FWL, Lasso, spectral cap), bootstrap coefficient bank, chronological residual pools, unconditional block sampler |
| `gaussian.py` | Gaussian conditional completion (joint path moments, Schur conditioning, mixture reweighting) + the Gaussian-law sampler used by the VAR/BVAR rows |
| `student_t.py` | Student-t smoothed residual-block bridge (SMC, 10,000 particles, CESS tempering, rejuvenation) |
| `gaussian_var.py` | dense Gaussian VAR benchmark |
| `bvar.py` | fixed-prior Minnesota BVAR benchmark (semi-conjugate Gibbs) |
| `ar_t.py` | univariate AR-t benchmark (BIC over lags {1,2}, conditional-t MLE) |
| `scoring.py` | fair CRPS, qWCRPS (US and UK tail maps), energy, variogram, coverage, interval widths, PIT-KS |
| `results.py` | CSV results writer: per-origin scores, thesis summary table, pairwise differences with moving-block bootstrap intervals |
| `data.py` | US Fed loader + preferred graph; UK8 loader; Conditions B target construction |

## Runners

- `uncon.py` -- unconditional, four generators.  US thesis Panel A: GIB-VAR
  0.6293, Gaussian VAR 0.7943, BVAR 0.6563, AR-t 0.6496.
  `--uk path/to/uk_history.csv`: GIB-VAR 0.8378, Gaussian VAR 1.1570,
  BVAR 0.9026, AR-t 0.8066 (thesis-reported Q4 subset, 8 origins).
- `cond.py` -- conditional (unemployment supplied), five rows.  US: Gaussian
  VAR 0.7338, AR-t 0.6507, Gaussian GIB 0.6002, Student-t GIB 0.6021,
  BVAR 0.6184.  UK via `--uk`: 1.0653 / 0.8741 / 0.9940 / 0.9784 / 0.9615.
- `narrative.py` -- Fed 2024 Conditions B, the three reported masks,
  Student-t vs BVAR from the 2023Q4 jump-off.

`uncon.py` and `cond.py` print the thesis table and write CSV results to
`results/<task>_<domain>/` (`--out` to change): `summary.csv` (the table),
`crps_by_origin.csv` (CRPS per origin per generator, crisis-overlap flag,
best generator), `scores_by_origin.csv` (every score per origin x seed x
generator) and `pairwise_differences.csv` (model A minus model B with the
95% moving-block bootstrap interval, overall / crisis / calm).  Each folder
carries a README explaining the files and how the interval is built
(`results.py` holds the code).
- `class_app/` -- the CLASS stress-testing application; see
  `class_app/README.md` for the archive setup and run order.

All input data lives in `data/` (see below).  Student-t rows move by about 0.005
across machines (SMC resampling amplifies floating-point noise); everything
else reproduces digit-for-digit under pinned dependencies (numpy 2.4.6,
pandas 2.3.3, scikit-learn 1.9.0, scipy 1.17.1).

## Data (`data/`)

    data/us/     2026_Final_Historic_Domestic.csv            main US history (committed)
                 2024-Table_2A_Historic_Domestic.csv          2024 vintage for the narrative study (committed)
                 2024-Table_Exploratory_Macro_Conditions_B_Domestic.csv   (committed)
    data/uk/     uk_macro_research16_historical.csv          licensed UK panel -- NOT committed; pass
                 source/                                     your copy to --uk, or drop it here
    data/class/  class8_protocol.json                        locked CLASS8 protocol (committed)
                 mpls_archive/                               public Minneapolis CLASS archive -- NOT
                                                             committed; see class_app/README.md

Every runner and the CLASS application read from this folder by default;
each also accepts explicit paths on the command line.
