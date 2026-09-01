# msc_code -- the report's methods

Everything runs from this folder.  Requirements: numpy, pandas,
scikit-learn, scipy (no package installation needed).  Reported numbers
reproduce digit-for-digit under pinned dependencies (numpy 2.4.6, pandas
2.3.3, scikit-learn 1.9.0, scipy 1.17.1), except the Student-t rows, which
move by a few thousandths across machines (SMC resampling amplifies
floating-point noise; documented in the report and in `src/README.md`).

## Layout

    src/          every model of the report in one short readable file each, plus
                  check_benchmarks.py (the US and UK rolling-origin evaluations, both
                  tasks) and check_narrative.py (the Conditions B narrative study)
    evaluation/   data loaders, scoring, CSV results writer
    class_app/    the CLASS stress-testing application (see class_app/README.md)
    data/         all input data;  results/   run outputs (gitignored)

## Start here: `src/`

| file | what it is | lines |
|---|---|---|
| `src/gibvar.py` | GIB-VAR: graph-weighted Lasso (FWL partialling, forward-validated penalty, three-parent refit), spectral cap, 25 block-bootstrap refits with chronological residual pools, unconditional block sampler | ~150 |
| `src/gaussian_adapter.py` | Gaussian completion: path mean and covariance, Schur conditioning, mixture reweighting; the unconditional Gaussian law used by the VAR and BVAR rows | ~80 |
| `src/student_t.py` | Student-t residual-block bridge: shrunk historical blocks, covariance-matched t noise, block likelihood, conditional t draw, SMC with resampling | ~160 |
| `src/gaussian_var.py` | dense Gaussian VAR benchmark | ~40 |
| `src/ar_t.py` | independent AR-t benchmark | ~110 |
| `src/bvar.py` | fixed-prior Minnesota BVAR benchmark (semi-conjugate Gibbs) | ~90 |
| `src/check_benchmarks.py` | fits all six models at every rolling origin, generates, scores and writes `results/<task>_<domain>/` | |
| `src/check_narrative.py` | the Fed 2024 Conditions B study: three masks, Gaussian GIB-VAR / Student-t GIB-VAR / BVAR from the 2023Q4 jump-off | |

    python src/check_benchmarks.py                        # US and UK, both tasks (~15 min)
    python src/check_benchmarks.py --domain US --limit 3  # smoke test
    python src/check_narrative.py                         # Conditions B masks (~1 min)

Expected rows -- US unconditional: GIB-VAR 0.6293, Gaussian VAR 0.7943, BVAR
0.6563, AR-t 0.6496; US conditional: Gaussian VAR 0.7338, AR-t 0.6507,
Gaussian GIB 0.6002, Student-t GIB 0.6021, BVAR 0.6184; UK: 0.8378 / 1.1570 /
0.9026 / 0.8066 and 1.0653 / 0.8741 / 0.9940 / 0.9784 / 0.9615.  `src/README.md`
lists the narrative rows and the one implementation difference behind the
Student-t noise.

## `class_app/`

The CLASS application (complete the DFAST 2020 severely adverse scenario with
Student-t GIB-VAR, Gaussian GIB-VAR and the BVAR, project every path through
the public Minneapolis CLASS model) runs on the `src/` models through
`class_app/_backend.py`.  See `class_app/README.md` for the archive setup and
run order.

## Results CSVs

Each results folder holds `summary.csv` (the table), `crps_by_origin.csv`
(CRPS per origin per generator, crisis-overlap flag, best generator),
`scores_by_origin.csv` (every score per origin x seed x generator),
`summary_by_regime.csv` and `pairwise_differences.csv` (model A minus model
B with the 95% moving-block bootstrap interval, overall / crisis / calm), plus
a README explaining the files and how the interval is built
(`evaluation/results.py` holds the code).

## Data (`data/`)

    data/us/     2026_Final_Historic_Domestic.csv            main US history (committed)
                 2024-Table_2A_Historic_Domestic.csv          2024 vintage for the narrative study (committed)
                 2024-Table_Exploratory_Macro_Conditions_B_Domestic.csv   (committed)
    data/uk/     uk_macro_research16_historical.csv          licensed UK panel -- NOT committed; pass
                 source/                                     your copy to check_benchmarks.py, or drop it here
    data/class/  class8_protocol.json                        locked CLASS8 protocol (committed)
                 mpls_archive/                               public Minneapolis CLASS archive -- NOT
                                                             committed; see class_app/README.md

Every runner and the CLASS application read from this folder by default;
`check_benchmarks.py` and `check_narrative.py` also accept explicit paths.
