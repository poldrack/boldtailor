"""Checkpointed fitting and recovery for a single paired simulation case."""

from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from filelock import FileLock

from experiments.hrf_estimation.splits import make_splits
from experiments.hrf_simulations.dataset import generate
from experiments.hrf_simulations.design import MODELS, task_design
from experiments.hrf_simulations.fitting import DesignCache, fit
from experiments.hrf_simulations.io import (
    atomic_npz,
    atomic_text,
    complete,
    file_hash,
    json_value,
    verify,
    write_json,
)
from experiments.hrf_simulations.metrics import TIMES, curve, library_floor, recovery
from experiments.hrf_simulations.truths import truth_library

_CACHE = DesignCache()


def implementation_fingerprint():
    root = Path(__file__).parent
    paths = [
        p
        for p in root.glob("*.py")
        if not p.name.startswith("test_")
        and p.name not in ("results.py", "reporting.py", "render.py")
    ]
    paths += [
        root.parent / "hrf_estimation" / name
        for name in (
            "inputs.py",
            "timing.py",
            "design.py",
            "reml.py",
            "reml_scalar.py",
            "reml_extensions.py",
            "regularized_noise.py",
            "scoring.py",
            "splits.py",
            "estimators.py",
        )
    ]
    paths += list((root.parents[1] / "src/boldtailor").glob("*.py"))
    digest = sha256()
    for path in sorted(paths):
        digest.update(str(path.relative_to(root.parents[1])).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _input_hash(runs, profile):
    digest = sha256()
    for run, donor in zip(runs, profile["runs"], strict=True):
        digest.update(run.run_id.encode())
        digest.update(run.events.to_csv(index=False).encode())
        for array in (
            run.frame_times,
            run.confounds.to_numpy(float),
            donor["residual"],
            donor["nuisance"],
        ):
            digest.update(np.asarray(array).tobytes())
        digest.update(json.dumps(list(run.confounds.columns)).encode())
    return digest.hexdigest()


def _splits(runs, enabled):
    all_ids = tuple(r.run_id for r in runs)
    evaluations = [("full", all_ids, all_ids)]
    if not enabled:
        return evaluations
    root = Path(__file__).parent.parent / "hrf_estimation"
    provenance = root / "reports/focused_comparison_provenance.json"
    matched = None
    if provenance.exists():
        for record in json.loads(provenance.read_text()):
            if record["method"] != "Original REML":
                continue
            manifest = root / record["source"] / "manifest.json"
            if manifest.exists():
                splits = json.loads(manifest.read_text())["splits"]
                if splits and set(splits[0]["train"] + splits[0]["test"]) == set(
                    all_ids
                ):
                    matched = [(s["train"], s["test"]) for s in splits[:2]]
                    break
    if matched is None:
        matched = [(s.train, s.test) for s in make_splits(all_ids, 20261110, 1)]
    return evaluations + [
        (f"half{i+1}", tuple(a), tuple(b)) for i, (a, b) in enumerate(matched)
    ]


def _reconstruction(runs, targets, candidate, model, convention):
    if candidate is None:
        return dict(task_reconstruction_rmse=np.nan, observed_r2=np.nan)
    error, count, sse, sst = 0.0, 0, 0.0, 0.0
    for run, target in zip(runs, targets, strict=True):
        parts = task_design(run, candidate, model, convention)
        matrix = np.column_stack([parts.task, parts.junk, parts.nuisance])
        beta = np.linalg.lstsq(matrix, run.signals[:, 0], rcond=None)[0]
        predicted = parts.task @ beta[: parts.task.shape[1]]
        error += np.sum((predicted - target) ** 2)
        count += len(target)
        sse += np.sum((run.signals[:, 0] - matrix @ beta) ** 2)
        sst += np.sum((run.signals[:, 0] - run.signals[:, 0].mean()) ** 2)
    return dict(
        task_reconstruction_rmse=float(np.sqrt(error / count)),
        observed_r2=float(1 - sse / sst) if sst > 0 else np.nan,
    )


def _fit_case(
    data, targets, truth, library, case, profile, models, evaluations, settings
):
    lookup = {r.run_id: r for r in data}
    signals = dict(zip(lookup, targets, strict=True))
    conventions = (
        ("brief", "stimulus")
        if case["mechanism"] == "duration"
        else (case["convention"],)
    )
    rows, curves, details = [], [], []
    fitted = {}
    estimators = (
        ("iid", "ar1", "ar1_oracle") if case["noise"] == "ar1" else ("iid", "ar1")
    )
    for convention in conventions:
        for model in models:
            # Pure RTDur columns do not depend on convention; still report both
            # labels for balanced comparisons, but reuse their identical fits.
            for estimator in estimators:
                for label, train_ids, test_ids in evaluations:
                    start = time.monotonic()
                    key = (
                        model,
                        None if model == "RTDur" else convention,
                        estimator,
                        label,
                    )
                    if key not in fitted:
                        fitted[key] = fit(
                            [lookup[k] for k in train_ids],
                            library,
                            model,
                            convention,
                            "iid" if estimator == "iid" else "ar1",
                            oracle_rho=(
                                profile["rho"] if estimator == "ar1_oracle" else None
                            ),
                            cache=_CACHE,
                            settings=settings,
                        )
                    result = fitted[key]
                    candidate = result.kernels[0]
                    score = _reconstruction(
                        [lookup[k] for k in test_ids],
                        [signals[k] for k in test_ids],
                        candidate,
                        model,
                        convention,
                    )
                    rho_true = (
                        profile["rho"]
                        if case["noise"] == "ar1"
                        else 0.0 if case["noise"] == "white" else np.nan
                    )
                    rows.append(
                        dict(
                            **{k: v for k, v in case.items() if k != "convention"},
                            generating_convention=case["convention"],
                            convention=convention,
                            model=model,
                            estimator=estimator,
                            split=label,
                            status=result.status[0],
                            rho=result.diagnostics["rho"],
                            rho_true=rho_true,
                            seconds=time.monotonic() - start,
                            **recovery(candidate, truth),
                            **score,
                        )
                    )
                    curves.append(
                        curve(candidate) if candidate else np.full(len(TIMES), np.nan)
                    )
                    details.append(result.diagnostics)
    return pd.DataFrame(rows), np.asarray(curves), details


def run_case(
    runs, library, profile, case, destination, models=MODELS, splits=True, settings=None
):
    folder = Path(destination)
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / ".lock")):
        return _run_locked(
            runs,
            library,
            profile,
            dict(case),
            folder,
            tuple(models),
            splits,
            settings or {},
        )


def _run_locked(runs, library, profile, case, folder, models, splits, settings):
    if "snr" not in case:
        case["snr"] = float(
            profile["snr_quantiles"][{"q25": 0, "q50": 1, "q75": 2}[case["snr_level"]]]
        )
    truth = truth_library(library)[case["truth"]]
    evaluations = _splits(runs, splits)
    manifest = dict(
        case=case,
        models=models,
        splits=evaluations,
        settings=settings,
        input_hash=_input_hash(runs, profile),
        library_fingerprint=library.fingerprint,
        implementation=implementation_fingerprint(),
        truth=dict(kind=truth.kind, parameters=truth.parameters, curve=curve(truth)),
        profile={k: v for k, v in profile.items() if k != "runs"},
        library_floor=library_floor(truth, library),
        conventions="physical unit-height boxcars; HRF unit peak; no event normalization",
        evaluation="OLS amplitudes refit on evaluation runs; HRF only held out for halves",
    )
    manifest = json.loads(json.dumps(manifest, default=json_value, sort_keys=True))
    previous = folder / "manifest.json"
    if previous.exists() and json.loads(previous.read_text()) != manifest:
        raise ValueError("manifest mismatch; choose a new output directory")
    if (folder / "COMPLETE.json").exists():
        verify(folder)
        return pd.read_csv(folder / "scores.tsv", sep="\t")
    write_json(previous, manifest)
    start = time.monotonic()
    data, targets, metadata = generate(runs, truth, profile, case)
    table, curves, details = _fit_case(
        data, targets, truth, library, case, profile, models, evaluations, settings
    )
    atomic_text(folder / "scores.tsv", table.to_csv(sep="\t", index=False))
    atomic_npz(folder / "hrfs.npz", times=TIMES, truth=curve(truth), curves=curves)
    metadata["seconds_total"] = time.monotonic() - start
    write_json(folder / "generation.json", metadata)
    write_json(folder / "fits.json", details)
    complete(
        folder,
        ["manifest.json", "scores.tsv", "hrfs.npz", "generation.json", "fits.json"],
    )
    return table
