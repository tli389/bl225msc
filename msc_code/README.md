# msc_code

Code for the report. Requirements: Python 3 with numpy, pandas, scikit-learn,
scipy. Run everything from this folder.

    src/          the models, one short file each: gibvar.py, gaussian_adapter.py,
                  student_t.py, gaussian_var.py, ar_t.py, bvar.py
    evaluation/   data loaders, scoring, CSV writer
    class_app/    the CLASS stress-testing application
    data/         inputs;  results/  outputs (gitignored)

## Rolling-origin evaluations (report tables) and the narrative study

    python src/check_benchmarks.py                        # US and UK, both tasks, ~15 min
    python src/check_benchmarks.py --domain US --limit 3  # quick smoke test
    python src/check_narrative.py                         # Conditions B masks, ~1 min

`check_benchmarks.py` writes `results/<task>_<domain>/` (`summary.csv` is the
report's table, plus per-origin scores and the pairwise bootstrap intervals)
and prints each generator's CRPS next to the report's. The UK panel
(`data/uk/uk_macro_research16_historical.csv`) contains licensed series and is
not included; pass your copy with `--uk path.csv` or drop it in place.

## CLASS application

Download the Minneapolis Fed "COVID-19 Stress Test Tool" archive
(https://www.minneapolisfed.org/banking/financial-studies-and-community-banking/covid-19-stress-test-tool)
and unzip it to `data/class/mpls_archive/` so that `macro_data_hist.csv`,
`macro_data_proj.csv`, `y9c_bhc_data.csv` and `output/estimated_model_coefficients.csv`
exist there. Then, in order:

    python class_app/run_matched_official.py     --out results/matched_official
    python class_app/run_generator_comparison.py --out results/generator_comparison
    python class_app/run_two_condition_sensitivity.py --out results/two_condition

The last step is the application in the report (unemployment and HPI growth
supplied, six variables completed). Steps 2 and 3 default to a small smoke
configuration (about three minutes each); add `--full` for the report protocol
(1,000 paths, 10,000 particles), which takes about an hour per step.

## Reproduction

All rows reproduce digit-for-digit except the Student-t GIB-VAR rows, which
land within a few thousandths: the runs behind the report added an adaptive
likelihood-tempering step when particle weights concentrated, `src/student_t.py`
resamples instead, and SMC resampling amplifies floating-point noise.
