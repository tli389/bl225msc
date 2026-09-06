# msc_code

Requires Python 3.10+ and NumPy, pandas, SciPy and scikit-learn.
Run these commands from this folder:

```powershell
python -m pip install numpy pandas scipy scikit-learn
python src/check_benchmarks.py --domain US
python src/check_narrative.py
python bvar_sensitivity/run_comparison.py
```

Inputs are in `data/`. Benchmark outputs go to `results/`; BVAR sensitivity
outputs go to `bvar_sensitivity/runs/`. Add `--limit 1` to the benchmark command
or `--years 2004 2004` to the sensitivity command for a quick check.

The UK dataset is not included because it contains licensed series.
For the UK run, provide a local copy at `data/uk/uk_macro_research16_historical.csv`
with a `quarter` column and the variables listed under `UK_FEATURES` in
[evaluation/data.py](evaluation/data.py), then run
`python src/check_benchmarks.py --domain UK`.

CLASS requires the [Minneapolis Fed archive](https://www.minneapolisfed.org/banking/financial-studies-and-community-banking/covid-19-stress-test-tool)
unpacked into `data/class/mpls_archive/`. Then run these in order:

```powershell
python class_app/run_matched_official.py --out results/matched_official
python class_app/run_generator_comparison.py --out results/generator_comparison --full
python class_app/run_two_condition_sensitivity.py --out results/two_condition --full
```

Omit `--full` for a quick CLASS check. The CLASS archive is not included.

Student-t GIB-VAR uses SMC, so results can vary slightly. Observed US
conditional CRPS has ranged from about **0.6009 to 0.6104**.
