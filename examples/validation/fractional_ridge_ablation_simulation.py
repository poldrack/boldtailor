"""Nested, leakage-safe ablations of fractional-ridge modeling choices."""

import argparse
from importlib.metadata import version
from itertools import product
import json
from pathlib import Path

import numpy as np
import pandas as pd

from boldtailor.fractional_ridge import fraction_grid
from boldtailor._single_trial_design import compile_trial_run
from boldtailor.single_trial import r_squared
from boldtailor.hrf_library import HrfLibrary
from boldtailor.trial_encoding import evaluate_trial_encoding
from examples.validation.fractional_ridge_ablation import prepare_runs, fit_split
from examples.validation.ridge_objective_simulation import score_prediction_runs

TRAIN, TEST = (0, 1, 2, 3), (4, 5)
FRACTIONS = (1.0, 0.8, 0.5, 0.2, 0.1)
FEATURES = ("strong_linear", "weak_linear", "nonlinear")
BASELINE = "candidate__normalized__per_run"
METRICS = (
    "beta_rmse",
    "within_run_beta_rmse",
    "amplitude_ratio",
    "within_run_amplitude_ratio",
    "within_run_correlation",
    "common_ols_r2",
    "latent_prediction_r2",
    "slope_rmse",
    "strongest_boundary",
    "is_ols",
)


def _noise(rng, shape, kind):
    values = rng.normal(size=shape)
    if kind == "none":
        return values * 0
    if kind == "white":
        return values * 0.2
    if kind != "ar_hetero":
        raise ValueError("noise must be none, white or ar_hetero")
    for t in range(1, len(values)):
        values[t] = 0.6 * values[t - 1] + 0.8 * values[t]
    return values * (0.6 * np.linspace(0.5, 1.5, len(values)))[:, None]


def _latent_betas(predictors, run, offset, rng):
    rt = predictors.response_time.to_numpy()
    trial_type = predictors.trial_type.to_numpy()
    signal = rt[:, None] * [1.2, 0.2, 0.8] + trial_type[:, None] * [-0.8, -0.15, -0.6]
    signal[:, 2] += 1.5 * (rt - 1.1) ** 2
    return signal + [2.0, 1.0, 2.0] + run * offset + rng.normal(0, 0.5, signal.shape)


def make_problem(seed, *, isi, durations, noise, hrf_mismatch, offset, n_trials=24):
    rng = np.random.default_rng(seed)
    result = {
        key: []
        for key in (
            "designs",
            "truth_designs",
            "signals",
            "truth",
            "predictors",
            "events",
        )
    }
    hrf = (
        HrfLibrary.from_parameters([[7, 16, 1.2, 1, 6, 0, 32]]).candidates[1]
        if hrf_mismatch
        else "spm"
    )
    for r in range(6):
        predictors = pd.DataFrame(
            dict(
                response_time=rng.uniform(0.3, 1.8, n_trials) + r * 0.15,
                trial_type=rng.integers(0, 2, n_trials),
            )
        )
        if durations not in ("uniform", "variable"):
            raise ValueError("durations must be uniform or variable")
        duration = (
            np.full(n_trials, 1.5)
            if durations == "uniform"
            else rng.choice([0.5, 1.5, 3.0], n_trials)
        )
        events = predictors.assign(
            onset=8 + np.arange(n_trials) * isi, duration=duration
        )
        times = 0.8 + 1.6 * np.arange(int(np.ceil((8 + n_trials * isi + 40) / 1.6)))
        confounds = pd.DataFrame(
            dict(
                motion=np.sin(np.arange(len(times)) / 8 + r),
                drift=np.linspace(-1, 1, len(times)),
            )
        )
        x, n, _ = compile_trial_run(events, times, confounds, f"run-{r}")
        truth_x, _, _ = compile_trial_run(events, times, confounds, f"run-{r}", hrf=hrf)
        truth = _latent_betas(predictors, r, offset, rng)
        y = 25 + truth_x.to_numpy() @ truth + _noise(rng, (len(times), 3), noise)
        y += confounds.motion.to_numpy()[:, None] * 0.3
        for key, value in dict(
            designs=(x.to_numpy(), n.to_numpy()),
            truth_designs=truth_x.to_numpy(),
            signals=y,
            truth=truth,
            predictors=predictors,
            events=events,
        ).items():
            result[key].append(value)
    return result


def _ratio(numerator, denominator):
    out = np.full_like(np.asarray(numerator, dtype=float), np.nan)
    return np.divide(numerator, denominator, out=out, where=denominator > 0)


def recovery_metrics(estimated, truth, predictions, ols_targets):
    estimated_all, truth_all = np.concatenate(estimated), np.concatenate(truth)
    ec = [e - e.mean(0) for e in estimated]
    tc = [t - t.mean(0) for t in truth]
    e, t = np.concatenate(ec), np.concatenate(tc)
    correlations = [
        _ratio(np.sum(a * b, 0), np.linalg.norm(a, axis=0) * np.linalg.norm(b, axis=0))
        for a, b in zip(ec, tc)
    ]
    return dict(
        beta_rmse=np.sqrt(np.mean((estimated_all - truth_all) ** 2, 0)),
        within_run_beta_rmse=np.sqrt(np.mean((e - t) ** 2, 0)),
        amplitude_ratio=_ratio(
            np.sum(estimated_all * truth_all, 0), np.sum(truth_all**2, 0)
        ),
        within_run_amplitude_ratio=_ratio(np.sum(e * t, 0), np.sum(t * t, 0)),
        within_run_correlation=np.mean(correlations, 0),
        common_ols_r2=score_prediction_runs(ols_targets, predictions, center=True)[
            "r2"
        ],
        latent_prediction_r2=score_prediction_runs(truth, predictions, center=True)[
            "r2"
        ],
    )


def _tune(prepared, predictors, fractions, target, scope):
    scores = []
    for fraction in fractions:
        fits = [
            fit_split(
                prepared,
                predictors,
                train=[r for r in TRAIN if r != v],
                test=[v],
                fraction=fraction,
                target=target,
                scope=scope,
            )
            for v in TRAIN
        ]
        scores.append(
            r_squared(
                np.sum([f["sse"] for f in fits], 0), np.sum([f["sst"] for f in fits], 0)
            )
        )
    scores = np.array(scores)
    if not np.isfinite(scores).all():
        raise ValueError("experiment produced undefined candidate scores")
    winners = np.argmax(scores >= scores.max(0) - 1e-12, axis=0)
    return scores, winners


def _candidate_rows(config, fractions, scores):
    return [
        dict(
            config=config, feature=FEATURES[v], fraction=f, inner_r2=float(scores[i, v])
        )
        for i, f in enumerate(fractions)
        for v in range(len(FEATURES))
    ]


def _selected_rows(
    fit,
    problem,
    reference_slopes,
    *,
    config,
    basis,
    target,
    scope,
    fraction,
    score,
    feature,
    strongest,
    calibrations=("none", "train_affine"),
):
    rows = []
    for calibration in calibrations:
        calibrated = calibration == "train_affine"
        betas = fit["calibrated_betas"] if calibrated else fit["test_betas"]
        predictions = (
            fit["calibrated_predictions"] if calibrated else fit["predictions"]
        )
        metrics = recovery_metrics(
            betas, [problem["truth"][r] for r in TEST], predictions, fit["ols_targets"]
        )
        scale, offset = fit["calibration"][:, feature] if calibrated else (1.0, 0.0)
        slope = fit["coefficients"][1:, feature] * scale
        achieved = fit["test_fractions"][:, feature]
        rows.append(
            dict(
                config=config,
                basis=basis,
                target=target,
                scope=scope,
                calibration=calibration,
                feature=FEATURES[feature],
                selected_fraction=float(fraction),
                inner_r2=float(score),
                scale=float(scale),
                offset=float(offset),
                training_alpha_mean=float(fit["training_alphas"][:, feature].mean()),
                test_alpha_mean=float(fit["test_alphas"][:, feature].mean()),
                test_fraction_mean=float(achieved.mean()),
                test_fraction_min=float(achieved.min()),
                test_fraction_max=float(achieved.max()),
                strongest_boundary=bool(strongest),
                is_ols=fraction == 1,
                slope_rmse=float(
                    np.sqrt(np.mean((slope - reference_slopes[:, feature]) ** 2))
                ),
                **{k: float(v[feature]) for k, v in metrics.items()},
            )
        )
    return rows


def evaluate_problem(problem, *, fractions=FRACTIONS):
    fractions = tuple(sorted(fraction_grid(fractions), reverse=True))
    prepared = {
        basis: prepare_runs(problem["designs"], problem["signals"], basis=basis)
        for basis in ("normalized", "raw")
    }
    # Truth is used only in reported metrics, never in fitting or selection.
    reference = evaluate_trial_encoding(
        problem["truth"], problem["predictors"], train_runs=TRAIN, test_runs=TEST
    ).coefficients[1:]
    rows, candidates = [], []
    for target, basis, scope in product(
        ("candidate", "fixed_ols"), ("normalized", "raw"), ("per_run", "pooled_train")
    ):
        config = "__".join((target, basis, scope))
        scores, winners = _tune(
            prepared[basis], problem["predictors"], fractions, target, scope
        )
        candidates.extend(_candidate_rows(config, fractions, scores))
        for winner in np.unique(winners):
            fraction = fractions[winner]
            fit = fit_split(
                prepared[basis],
                problem["predictors"],
                train=TRAIN,
                test=TEST,
                fraction=fraction,
                target=target,
                scope=scope,
            )
            for v in np.flatnonzero(winners == winner):
                rows.extend(
                    _selected_rows(
                        fit,
                        problem,
                        reference,
                        config=config,
                        basis=basis,
                        target=target,
                        scope=scope,
                        fraction=fraction,
                        score=scores[winner, v],
                        feature=v,
                        strongest=winner == len(fractions) - 1,
                    )
                )
    ols = fit_split(
        prepared["raw"],
        problem["predictors"],
        train=TRAIN,
        test=TEST,
        fraction=1.0,
        target="fixed_ols",
        scope="per_run",
    )
    for v in range(len(FEATURES)):
        rows.extend(
            _selected_rows(
                ols,
                problem,
                reference,
                config="ols_reference",
                basis="raw",
                target="fixed_ols",
                scope="per_run",
                fraction=1.0,
                score=np.nan,
                feature=v,
                strongest=False,
                calibrations=("none",),
            )
        )
    return rows, candidates


def scenarios():
    keys = ("isi", "durations", "noise", "hrf_mismatch", "offset")
    return [
        dict(zip(keys, values))
        for values in product(
            (4.0, 12.0),
            ("uniform", "variable"),
            ("white", "ar_hetero"),
            (False, True),
            (0.0, 1.0),
        )
    ]


def _aggregate(seed_means):
    grouped = seed_means.groupby(["config", "calibration"])
    summary = grouped[list(METRICS)].agg(["mean", "sem"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary["seed_count"] = grouped.size()
    return summary.reset_index()


def save_summaries(frame, output):
    seed_means = (
        frame.groupby(["seed", "config", "calibration"])[list(METRICS)]
        .mean()
        .reset_index()
    )
    seed_means.to_csv(output / "seed-means.csv", index=False)
    _aggregate(seed_means).to_csv(output / "summary.csv", index=False)
    baseline = seed_means[
        (seed_means.config == BASELINE) & (seed_means.calibration == "none")
    ].set_index("seed")
    paired = seed_means.copy()
    for m in METRICS:
        paired[m] -= paired.seed.map(baseline[m])
    paired.to_csv(output / "paired-seed-deltas.csv", index=False)
    _aggregate(paired).to_csv(output / "paired-summary.csv", index=False)
    groups = [
        "seed",
        "isi",
        "durations",
        "noise",
        "hrf_mismatch",
        "offset_increment",
        "feature",
        "config",
        "calibration",
    ]
    frame.groupby(groups)[list(METRICS)].mean().to_csv(
        output / "stratified-seed-means.csv"
    )


def run_simulation(
    *, seeds, output_dir, scenarios=None, n_trials=24, fractions=FRACTIONS
):
    if isinstance(seeds, bool) or not isinstance(seeds, int) or seeds < 1:
        raise ValueError("seeds must be a positive integer")
    settings = scenarios if scenarios is not None else globals()["scenarios"]()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    all_rows, all_candidates = [], []
    for seed in range(seeds):
        for number, setting in enumerate(settings):
            problem = make_problem(seed, **setting, n_trials=n_trials)
            rows, candidates = evaluate_problem(problem, fractions=fractions)
            context = dict(seed=seed, scenario=number, **setting)
            context["offset_increment"] = context.pop("offset")
            all_rows.extend(dict(context, **r) for r in rows)
            all_candidates.extend(dict(context, **r) for r in candidates)
        print(f"Completed seed {seed+1}/{seeds}", flush=True)
    frame = pd.DataFrame(all_rows)
    frame.to_csv(output / "selections.csv.gz", index=False)
    pd.DataFrame(all_candidates).to_csv(output / "candidates.csv.gz", index=False)
    save_summaries(frame, output)
    metadata = dict(
        seeds=list(range(seeds)),
        scenarios=settings,
        n_trials=n_trials,
        features=FEATURES,
        train_runs=TRAIN,
        test_runs=TEST,
        fractions=fractions,
        production_defaults_changed=False,
        pooled_alpha="estimated_on_training_runs_and_frozen_for_held_out_runs",
        calibration="post_selection_affine_fit_to_training_OLS_only",
        uncertainty="paired_scenario_and_feature_averages_within_seed_then_SEM_across_seeds",
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
    args = parser.parse_args()
    run_simulation(seeds=args.seeds, output_dir=args.output_dir)
