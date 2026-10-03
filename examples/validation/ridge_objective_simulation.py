"""Compare historical CV objectives with production within-run encoding.

Run with uv run python examples/validation/ridge_objective_simulation.py
--seeds 10 --output-dir /private/tmp/boldtailor-ridge-objectives.
"""

import argparse
from itertools import product
import json
from pathlib import Path
from importlib.metadata import version

import numpy as np
import pandas as pd

from boldtailor.single_trial import compile_trial_run, r_squared
from boldtailor.trial_encoding import evaluate_trial_encoding
from tests.oracles import fraction_beta_path, trial_beta_path

ALPHAS = (0.0, 0.1, 1.0, 10.0, 100.0, 1e4, 1e8)
FRACTIONS = (1.0, 0.8, 0.5, 0.2, 0.1)
OBJECTIVES = ("current", "centered", "fixed_ols", "within_run")
TRAIN, TEST = (0, 1, 2, 3), (4, 5)


def score_prediction_runs(targets, predictions, *, center):
    """Pool complete observation pairs; optionally remove test residual means."""
    n_features = targets[0].shape[1]
    sse, sst = np.zeros(n_features), np.zeros(n_features)
    offsets = []
    for target, prediction in zip(targets, predictions, strict=True):
        if target.shape != prediction.shape:
            raise ValueError("target and prediction shapes must match")
        offset = np.full(n_features, np.nan)
        for feature in range(n_features):
            valid = np.isfinite(target[:, feature]) & np.isfinite(
                prediction[:, feature]
            )
            if valid.sum() < 2:
                sse[feature] = sst[feature] = np.nan
                continue
            observed = target[valid, feature]
            residual = observed - prediction[valid, feature]
            offset[feature] = residual.mean() if center else 0.0
            sse[feature] += np.sum((residual - offset[feature]) ** 2)
            sst[feature] += np.sum((observed - observed.mean()) ** 2)
        offsets.append(offset)
    return dict(r2=r_squared(sse, sst), sse=sse, sst=sst, offsets=np.array(offsets))


def evaluate_candidate(betas, predictors, train, test, objective, ols_runs):
    if objective not in OBJECTIVES:
        raise ValueError("unknown experimental objective")
    fitted = evaluate_trial_encoding(
        betas,
        predictors,
        train_runs=train,
        test_runs=test,
        encoding_mode="within_run" if objective == "within_run" else "absolute",
    )
    targets = tuple((ols_runs if objective == "fixed_ols" else betas)[r] for r in test)
    score = score_prediction_runs(
        targets, fitted.predictions, center=objective in ("centered", "within_run")
    )
    return dict(
        score,
        targets=targets,
        predictions=fitted.predictions,
        coefficients=fitted.coefficients,
    )


def _problem(seed, isi, duration, noise, beta_noise, offset):
    rng = np.random.default_rng(seed)
    predictors, true, designs, signals = [], [], [], []
    times = 0.8 + 1.6 * np.arange(int(np.ceil((8 + 40 * isi + 40) / 1.6)))
    for run in range(6):
        p = pd.DataFrame(
            dict(
                response_time=rng.uniform(0.3, 1.8, 40),
                trial_type=rng.integers(0, 2, 40),
            )
        )
        events = p.assign(onset=8 + np.arange(40) * isi, duration=duration)
        nuisance = pd.DataFrame(
            dict(
                motion=np.sin(np.arange(len(times)) / 8 + run),
                drift=np.linspace(-1, 1, len(times)),
            )
        )
        x, n, _ = compile_trial_run(events, times, nuisance, f"run-{run}")
        beta = (
            2
            + run * offset
            + 0.8 * p.response_time
            - 0.6 * p.trial_type
            + rng.normal(0, beta_noise, 40)
        ).to_numpy()[:, None]
        y = 25 + x.to_numpy() @ beta + rng.normal(0, noise, (len(times), 1))
        predictors.append(p)
        true.append(beta)
        designs.append((x.to_numpy(), n.to_numpy()))
        signals.append(y)
    return predictors, true, designs, signals


def _paths(designs, signals):
    alpha_runs, fraction_runs = [], []
    for (x, n), y in zip(designs, signals, strict=True):
        alpha_runs.append([b for _, b in trial_beta_path(x, n, y, alphas=ALPHAS)])
        fraction_runs.append(
            [b for _, b, _ in fraction_beta_path(x, n, y, fractions=FRACTIONS)]
        )
    return dict(alpha=list(zip(*alpha_runs)), fraction=list(zip(*fraction_runs)))


def _inner_scores(candidates, predictors, objective, ols):
    scores = []
    for betas in candidates:
        losses, totals = [], []
        for validation in TRAIN:
            train = [r for r in TRAIN if r != validation]
            result = evaluate_candidate(
                betas, predictors, train, [validation], objective, ols
            )
            losses.append(result["sse"])
            totals.append(result["sst"])
        scores.append(
            float(r_squared(np.sum(losses, axis=0), np.sum(totals, axis=0))[0])
        )
    return np.array(scores)


def _projection(estimated, truth):
    norm = np.sum(truth**2)
    return float(np.sum(estimated * truth) / norm) if norm > 0 else float("nan")


def _correlation(estimated, truth):
    if np.ptp(estimated) == 0 or np.ptp(truth) == 0:
        return float("nan")
    return float(np.corrcoef(estimated, truth)[0, 1])


def _recovery_metrics(betas, true):
    estimated = np.concatenate([betas[r] for r in TEST])
    truth = np.concatenate([true[r] for r in TEST])
    centered = np.concatenate([betas[r] - betas[r].mean(0) for r in TEST])
    centered_true = np.concatenate([true[r] - true[r].mean(0) for r in TEST])
    correlations = [_correlation(betas[r][:, 0], true[r][:, 0]) for r in TEST]
    return dict(
        within_run_beta_rmse=float(np.sqrt(np.mean((centered - centered_true) ** 2))),
        within_run_amplitude_ratio=_projection(centered, centered_true),
        beta_rmse=float(np.sqrt(np.mean((estimated - truth) ** 2))),
        amplitude_ratio=_projection(estimated, truth),
        norm_ratio=float(np.linalg.norm(estimated) / np.linalg.norm(truth)),
        within_run_correlation=float(np.mean(correlations)),
    )


def _selected_row(betas, true, predictors, objective, ols, grid, winner, scores):
    result = evaluate_candidate(betas, predictors, TRAIN, TEST, objective, ols)
    current = evaluate_candidate(betas, predictors, TRAIN, TEST, "current", ols)
    return dict(
        selected_value=grid[winner],
        strongest_boundary=winner == len(grid) - 1,
        is_ols=winner == 0,
        inner_r2=float(scores[winner]),
        outer_r2=float(result["r2"][0]),
        current_objective_outer_r2=float(current["r2"][0]),
        outer_prediction_mse=float(result["sse"][0] / 80),
        removed_test_offset_mean=float(result["offsets"].mean()),
        train_runs=",".join(map(str, TRAIN)),
        test_runs=",".join(map(str, TEST)),
        **_recovery_metrics(betas, true),
    )


def scenario_rows(seed, isi, duration, noise, beta_noise, offset):
    predictors, true, designs, signals = _problem(
        seed, isi, duration, noise, beta_noise, offset
    )
    paths = _paths(designs, signals)
    ols = paths["alpha"][0]
    setting = dict(
        seed=seed,
        isi=isi,
        duration=duration,
        noise=noise,
        beta_noise=beta_noise,
        offset=offset,
    )
    rows, candidates = [], []
    for regularizer, grid in (("alpha", ALPHAS), ("fraction", FRACTIONS)):
        for objective in OBJECTIVES:
            scores = _inner_scores(paths[regularizer], predictors, objective, ols)
            if not np.isfinite(scores).all():
                raise ValueError("simulation produced undefined candidate scores")
            winner = int(np.flatnonzero(scores >= scores.max() - 1e-12)[0])
            selected = _selected_row(
                paths[regularizer][winner],
                true,
                predictors,
                objective,
                ols,
                grid,
                winner,
                scores,
            )
            rows.append(
                dict(setting, regularizer=regularizer, objective=objective, **selected)
            )
            candidates.extend(
                dict(
                    setting,
                    regularizer=regularizer,
                    objective=objective,
                    value=value,
                    inner_r2=float(score),
                )
                for value, score in zip(grid, scores, strict=True)
            )
    return rows, candidates


def confounding_stress():
    """Noiseless encoding stress test: x mean shifts 10/run, y offsets 0 or 20/run."""
    predictors = [pd.DataFrame(dict(value=np.arange(5.0) + 10 * r)) for r in range(6)]
    rows = []
    for offset in (0.0, 20.0):
        betas = [
            (3 * p.value.to_numpy() + r * offset)[:, None]
            for r, p in enumerate(predictors)
        ]
        for objective in ("current", "centered", "within_run"):
            fit = evaluate_candidate(betas, predictors, TRAIN, TEST, objective, betas)
            rows.append(
                dict(
                    objective=objective,
                    offset_increment=offset,
                    predictor_mean_increment=10.0,
                    true_slope=3.0,
                    slope=float(fit["coefficients"][1, 0]),
                    outer_r2=float(fit["r2"][0]),
                )
            )
    return rows


def run_simulation(*, seeds, output_dir):
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1:
        raise ValueError("seeds must be a positive count")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, candidates = [], []
    for seed in range(seeds):
        for values in product(
            (4.0, 14.3), (1.5, 3.0), (0.0, 0.3, 1.0), (0.0, 0.5), (0.0, 1.0)
        ):
            selected, scored = scenario_rows(seed, *values)
            rows.extend(selected)
            candidates.extend(scored)
        print(f"Completed seed {seed+1}/{seeds}", flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "selections.csv", index=False)
    pd.DataFrame(confounding_stress()).to_csv(
        output / "confounding-stress.csv", index=False
    )
    pd.DataFrame(candidates).to_csv(output / "candidates.csv", index=False)
    grouping = [
        "isi",
        "duration",
        "noise",
        "beta_noise",
        "offset",
        "regularizer",
        "objective",
    ]
    metrics = [
        "beta_rmse",
        "within_run_beta_rmse",
        "within_run_amplitude_ratio",
        "amplitude_ratio",
        "norm_ratio",
        "within_run_correlation",
        "outer_r2",
        "strongest_boundary",
        "outer_prediction_mse",
    ]
    summary = frame.groupby(grouping)[metrics].agg(["mean", "std", "sem", "count"])
    summary.to_csv(output / "summary.csv")
    metadata = dict(
        seeds=list(range(seeds)),
        train_runs=TRAIN,
        test_runs=TEST,
        alphas=ALPHAS,
        fractions=FRACTIONS,
        n_trials=40,
        tr=1.6,
        objectives=OBJECTIVES,
        scenarios_per_seed=48,
        default_changed=True,
        production_encoding_mode="within_run",
        historical_current="2026-09-27 shared-intercept absolute objective",
        encoding_objective_version=2,
        versions={
            name: version(name) for name in ("boldtailor", "numpy", "scipy", "nilearn")
        },
    )
    (output / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    run_simulation(seeds=arguments.seeds, output_dir=arguments.output_dir)
