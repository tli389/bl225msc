"""Run the US unconditional and conditional BVAR prior sensitivity."""

import os
for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from src.bvar import fit_minnesota_bvar
from src.gaussian_adapter import sample_gaussian_law, complete
from src.check_benchmarks import stable_seed
from evaluation.data import FEATURES, load_us_history
from evaluation.scoring import fair_crps
from bvar_sensitivity.glp_model import fit_conjugate_bvar

HORIZON, N_PATHS, N_DRAWS, SEEDS = 12, 1000, 100, (7, 19, 43)
MODELS = ("Existing fixed BVAR", "Conjugate fixed lambda", "GLP-inspired MAP lambda")
METRICS = ("crps", "coverage80", "coverage95", "width80_std", "width95_std", "mean_forecast_rmse_std")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def score(paths, truth, std, features):
    if not all(np.isfinite(a).all() for a in (paths, truth, std)) or np.any(std <= 0):
        raise ValueError("Scoring requires finite arrays and positive training scales")
    lo80, hi80, lo95, hi95 = np.percentile(paths, [10, 90, 2.5, 97.5], axis=0)
    errors = (paths.mean(axis=0) - truth) / std
    cells = []
    for h in range(len(truth)):
        for j, feature in enumerate(features):
            cells.append({
                "horizon": h + 1, "feature": feature,
                "crps": float(fair_crps(paths[:, h, j], truth[h, j]) / std[j]),
                "coverage80": float(lo80[h, j] <= truth[h, j] <= hi80[h, j]),
                "coverage95": float(lo95[h, j] <= truth[h, j] <= hi95[h, j]),
                "width80_std": float((hi80[h, j] - lo80[h, j]) / std[j]),
                "width95_std": float((hi95[h, j] - lo95[h, j]) / std[j]),
                "mean_forecast_error_std": float(errors[h, j]),
            })
    result = {key: float(np.mean([cell[key] for cell in cells])) for key in METRICS[:-1]}
    result[METRICS[-1]] = float(np.sqrt(np.mean(errors ** 2)))
    return result, cells


def run(years, out):
    data_path = ROOT / "data/us/2026_Final_Historic_Domestic.csv"
    history = load_us_history(str(data_path))
    values = history[FEATURES].to_numpy(float)
    ci = FEATURES.index("unemployment")
    hidden = [j for j in range(len(FEATURES)) if j != ci]
    origins = [i for i, q in enumerate(history.index)
               if q.quarter == 4 and years[0] <= q.year <= years[1] and i + HORIZON < len(history)]
    if not origins or origins[0] + 1 < 60:
        raise ValueError("Require Q4 origins with at least 60 training quarters and 12 future quarters")
    sources = [ROOT / p for p in ("src/bvar.py", "src/gaussian_adapter.py", "src/check_benchmarks.py",
               "evaluation/data.py", "evaluation/scoring.py", "bvar_sensitivity/glp_model.py",
               "bvar_sensitivity/run_comparison.py", "data/us/2026_Final_Historic_Domestic.csv")]
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    manifest = dict(status="running", models=MODELS, features=FEATURES,
                    origins=[str(history.index[i]) for i in origins], horizon=HORIZON,
                    paths=N_PATHS, draws=N_DRAWS, seeds=SEEDS, conditions=["unemployment"],
                    clipping=False, source_sha256=hashes, python=sys.version,
                    numpy=np.__version__, pandas=pd.__version__)
    write_json(out / "manifest.json", manifest)
    rows, cells, fits = [], [], []
    start = time.perf_counter()
    for oi, origin in enumerate(origins):
        name = str(history.index[origin])
        # Fit using the expanding history through this origin.
        train = values[:origin + 1]
        truth, x0 = values[origin + 1:origin + 1 + HORIZON], values[origin]
        std = train.std(axis=0, ddof=1)
        supplied = np.zeros_like(truth)
        supplied[:, ci] = truth[:, ci]
        for model in MODELS:
            if model == MODELS[0]:
                bank = fit_minnesota_bvar(train, stable_seed(211, "bvar_fit", name),
                                         n_draws=N_DRAWS, burn_in=150, thin=3,
                                         max_radius=0.995, max_iter=20000)
                diagnostic = {"lambda": None, "attempted_draws": None}
            else:
                bank, diagnostic = fit_conjugate_bvar(
                    train, stable_seed(211, "conjugate_bvar_fit", name),
                    optimize=model == MODELS[2], n_draws=N_DRAWS)
            if len(bank) != N_DRAWS:
                raise RuntimeError("The fitted parameter bank is incomplete")
            fits.append(dict(model=model, origin=name, **diagnostic))
            for seed in SEEDS:
                # Generate both tasks with the same Gaussian adapters as the main BVAR.
                unconditional = sample_gaussian_law(
                    bank, N_PATHS, HORIZON,
                    np.random.default_rng(stable_seed(seed, "unconditional", "minnesota_bvar", name)), x0)
                conditional, _ = complete(
                    bank, supplied, [ci], N_PATHS,
                    np.random.default_rng(stable_seed(seed, "conditional", "minnesota_bvar", name)), x0)
                error = np.max(np.abs(conditional[:, :, ci] - truth[:, ci]))
                if not np.isfinite(error) or error >= 1e-10:
                    raise RuntimeError("Conditional draws did not restore the supplied path")
                # Score all unconditional cells and only hidden conditional cells.
                for task, paths, target, scales, features in (
                    ("unconditional", unconditional, truth, std, FEATURES),
                    ("conditional", np.delete(conditional, ci, axis=2), np.delete(truth, ci, axis=1),
                     np.delete(std, ci), [FEATURES[j] for j in hidden]),
                ):
                    metrics, cell_scores = score(paths, target, scales, features)
                    identity = dict(model=model, scenario=task, origin=name, seed=seed)
                    rows.append(identity | metrics)
                    cells.extend(identity | cell for cell in cell_scores)
        print(f"[{time.perf_counter()-start:.1f}s] US {name} ({oi+1}/{len(origins)})", flush=True)

    # Average cells within seeds, seeds within origins, then origins.
    frame = pd.DataFrame(rows)
    by_origin = frame.groupby(["model", "scenario", "origin"], sort=False)[list(METRICS)].mean().reset_index()
    summary = by_origin.groupby(["model", "scenario"], sort=False)[list(METRICS)].mean().reset_index()
    for filename, table in (("per_seed_origin", frame), ("per_origin", by_origin), ("summary", summary),
                            ("per_cell", pd.DataFrame(cells)), ("fit_diagnostics", pd.DataFrame(fits))):
        table.to_csv(out / f"{filename}.csv", index=False)
    if any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[str(p.relative_to(ROOT))] for p in sources):
        raise RuntimeError("Source or data changed during the run")
    manifest.update(status="completed", elapsed_seconds=time.perf_counter() - start)
    write_json(out / "manifest.json", manifest)
    print(summary.pivot(index="model", columns="scenario", values="crps").reindex(MODELS).round(4))
    print(f"Results: {out}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs=2, type=int, default=[2004, 2022])
    parser.add_argument("--output", type=Path,
                        default=HERE / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    args = parser.parse_args()
    if args.years[0] > args.years[1]:
        parser.error("--years must be in increasing order")
    out = args.output.resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f"Output must be a new or empty directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    try:
        run(args.years, out)
    except Exception as error:
        path = out / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(path, manifest)
        raise
