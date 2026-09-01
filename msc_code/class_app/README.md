# class_app -- the CLASS stress-testing application

Runs the report's supervisory application: complete the DFAST 2020 severely
adverse scenario with Student-t GIB-VAR, Gaussian GIB-VAR and the Minnesota
BVAR, then project every completed path through the public Minneapolis
modification of the CLASS model (fixed 2019Q4 bank state, fixed coefficients).

`minneapolis_class.py` is the CLASS model, `adapter.py` maps CLASS8 paths to
CLASS inputs, and `_local.py` wires the three runners to the generators in
`../src/` (`_backend.py` provides the names and call signatures the runners
were written against on top of `src/gibvar.py`, `src/bvar.py`,
`src/gaussian_adapter.py` and `src/student_t.py`).  No package installation
is needed; everything runs from this folder.

## One-off setup: the Minneapolis archive

Download the "COVID-19 Stress Test Tool" zip from the Minneapolis Fed
(https://www.minneapolisfed.org/banking/financial-studies-and-community-banking/covid-19-stress-test-tool)
and unzip it so that these files exist:

    msc_code/data/class/mpls_archive/macro_data_hist.csv
    msc_code/data/class/mpls_archive/macro_data_proj.csv
    msc_code/data/class/mpls_archive/y9c_bhc_data.csv
    msc_code/data/class/mpls_archive/output/estimated_model_coefficients.csv
    (plus the other archive files)

`data/class/class8_protocol.json` (the locked CLASS8 protocol) ships with the
repository.  The archive itself is not committed.

## Run order (from the `msc_code` folder)

    python class_app/run_matched_official.py    --out results/matched_official
    python class_app/run_generator_comparison.py --out results/generator_comparison
    python class_app/run_two_condition_sensitivity.py --out results/two_condition

1. `run_matched_official.py` passes the published severely adverse path
   through CLASS with the revised 2019 seed history.  It takes seconds and
   reproduces the report's "Complete path" column exactly (minimum CET1
   8.25%, drawdown 299.7bn, NCO 316.7bn, provisions 406.2bn, PPNR 239.9bn,
   net income -125.9bn).
2. `run_generator_comparison.py` conditions on the unemployment path only
   and projects the three completions through CLASS.
3. `run_two_condition_sensitivity.py` is the application reported in the
   thesis: unemployment and rebased HPI growth supplied, six variables
   completed.  It reads the outputs of steps 1 and 2 (override with
   `--matched-reference`, `--matched-reference-manifest`,
   `--unemployment-only-dir`).

Steps 2 and 3 default to a **smoke** configuration (4 fitted systems, 48
paths, 128 particles; about three minutes each) that checks the plumbing.
Add `--full` for the report protocol (25 systems, 1,000 paths, 10,000
particles, 100 BVAR draws), which takes on the order of an hour or more.
With `--full`, the BVAR row and the complete-path column reproduce the
report's CLASS tables digit-for-digit; the Student-t row lands within its SMC
noise (see `src/README.md`).
