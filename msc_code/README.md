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
not included. To rebuild it, take the historical section of the Bank of
England's 2026 ICAAP scenario workbook (Bank Capital Stress Test resources on
bankofengland.co.uk) and write a quarterly CSV, 2001Q1 onward, with a `quarter`
column (e.g. `2001Q1`) and these columns in percentage units:
`real_gdp_growth_annualized_pct`, `unemployment_rate_pct`, `bank_rate_pct`,
`gilt_10y_pct`, `ig_corporate_spread_pct`, `hpi_qoq_growth_pct`,
`equity_qoq_growth_pct`, `cpi_inflation_yoy_pct` (the transformations in the
report's variable table). Pass it with `--uk path.csv` or drop it in place.

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
land within about 0.01: the runs behind the report added an adaptive
likelihood-tempering step when particle weights concentrated, `src/student_t.py`
resamples instead, and SMC resampling amplifies floating-point noise. The row
also varies across machines (0.6009-0.6104 observed for the US conditional
task), almost entirely at origins where the particle system degenerates; the
rankings are unchanged.
