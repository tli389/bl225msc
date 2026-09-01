# src -- the report's models

One short file per component of the report, written to follow the Proposed
Approach chapter step by step so the whole method can be read in one sitting.
`check_benchmarks.py` runs the rolling-origin evaluations from these files and
`check_narrative.py` the Conditions B narrative study; both print their rows
next to the report's.

| file | what it is | lines |
|---|---|---|
| `gibvar.py` | GIB-VAR backbone: graph-weighted Lasso with FWL partialling, forward-validated penalty, three-parent refit, spectral cap, 25 block-bootstrap refits with chronological residual pools, unconditional block sampler | ~150 |
| `gaussian_adapter.py` | Gaussian completion: path mean and covariance (mu, P, Omega), Schur conditioning (drag / shrink), mixture reweighting; plus the unconditional Gaussian law used by the VAR and BVAR rows | ~80 |
| `student_t.py` | Student-t residual-block bridge: shrunk historical blocks, covariance-matched t noise, block likelihood, conditional t draw, SMC with resampling | ~160 |
| `gaussian_var.py` | dense Gaussian VAR: OLS, cap and re-centre, 5% diagonal ridge | ~40 |
| `ar_t.py` | independent AR-t: AR(1)/AR(2) by conditional MLE, BIC on the common sample, independent t shocks | ~110 |
| `bvar.py` | Minnesota BVAR: standardised regression, the 0.9 / 0.2 / 0.1 / 10 / IW(K+2, I) prior, semi-conjugate Gibbs, burn-in, thinning, stability rejection | ~90 |
| `check_benchmarks.py` | the rolling-origin runner: fits all six models at every origin, generates, scores, writes `results/<task>_<domain>/` | |
| `check_narrative.py` | the Conditions B study: three masks, Gaussian GIB-VAR / Student-t GIB-VAR / BVAR from the 2023Q4 jump-off | |

## Run (from the `msc_code` folder)

    python src/check_benchmarks.py                        # US and UK, both tasks (~15 min)
    python src/check_benchmarks.py --domain US --limit 3  # smoke test
    python src/check_narrative.py                         # Conditions B masks (~1 min)

`check_benchmarks.py` imports only this folder and the loaders, scorer and CSV
writer in `../evaluation`.  At every rolling origin it fits all six models,
generates 1,000 paths under seeds {7, 19, 43}, scores them, and writes
`results/<task>_<domain>/` -- `summary.csv` (the report's table),
`crps_by_origin.csv`, `scores_by_origin.csv`, `summary_by_regime.csv` and
`pairwise_differences.csv`.  The UK run needs the licensed UK panel at its
default location (see the main README).

Expected rows:

    US  unconditional  GIB-VAR 0.6293   Gaussian VAR 0.7943   BVAR 0.6563   AR-t 0.6496
    US  conditional    Gaussian VAR 0.7338   AR-t 0.6507   Gaussian GIB-VAR 0.6002   Student-t 0.6021   BVAR 0.6184
    UK  unconditional  GIB-VAR 0.8378   Gaussian VAR 1.1570   BVAR 0.9026   AR-t 0.8066
    UK  conditional    Gaussian VAR 1.0653   AR-t 0.8741   Gaussian GIB-VAR 0.9940   Student-t 0.9784   BVAR 0.9615

    narrative  activity-labour       Gaussian GIB 1.1391   Student-t 1.1435   BVAR 0.9406
               property              Gaussian GIB 0.7220   Student-t 0.7057   BVAR 0.8825
               high-rate recession   Gaussian GIB 1.1774   Student-t 1.1698   BVAR 1.1792

Every row reproduces digit-for-digit except the Student-t rows, which land
within a few thousandths.  The runs behind the report's Student-t rows added
an adaptive likelihood-tempering step (with a rejuvenation move) whenever the
particle weights concentrated within a stage; `student_t.py` handles weight
concentration by resampling alone, and SMC resampling amplifies floating-point
noise, so those rows agree to within SMC noise rather than exactly.  Two AR-t
implementation details are kept because the fit is otherwise not identical:
the series is standardised before optimisation, and L-BFGS-B is started from
three degrees-of-freedom values.
