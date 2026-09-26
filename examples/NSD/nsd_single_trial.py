"""Fit independent NSD trial betas with canonical-HRF OLS and optional fixed ridge.

Run ``uv run python examples/NSD/nsd_single_trial.py --ridge-alpha 0.1``
from the repository root. RT is used only for descriptive checks after fitting.
"""

import argparse
from dataclasses import dataclass
from importlib.metadata import version
from numbers import Integral
from pathlib import Path
import re

import nibabel as nib
import numpy as np
import pandas as pd

from boldtailor.data import from_arrays
from boldtailor.single_trial import fit_single_trials
from boldtailor._single_trial_fit import validate_alpha
from boldtailor.publication import publish_artifact_set

if __package__:
    from .nsd_cifti import (
        BIDS_ROOT,
        MODEL,
        RunInputs,
        _input_paths,
        _sources,
        discover_runs,
        load_inputs,
    )
    from .rt_diagnostics import correlate_rt, scatter_artifact, select_vertices
    from .single_trial_artifacts import (
        all_model_paths,
        diagnostic_paths,
        json_artifact,
        model_paths,
        selected_vertex_table,
        single_trial_artifacts,
        table_artifact,
    )
else:
    from nsd_cifti import (
        BIDS_ROOT,
        MODEL,
        RunInputs,
        _input_paths,
        _sources,
        discover_runs,
        load_inputs,
    )
    from rt_diagnostics import correlate_rt, scatter_artifact, select_vertices
    from single_trial_artifacts import (
        all_model_paths,
        diagnostic_paths,
        json_artifact,
        model_paths,
        selected_vertex_table,
        single_trial_artifacts,
        table_artifact,
    )


@dataclass(frozen=True)
class TrialRun:
    inputs: RunInputs
    image: nib.Cifti2Image
    events: pd.DataFrame
    confounds: pd.DataFrame
    frame_times: np.ndarray
    label: str
    number: int


def _load_runs(inputs):
    runs = []
    for item in inputs:
        image, events, confounds, times = load_inputs(item)
        label = item.stem.rsplit("_", 1)[-1]
        runs.append(
            TrialRun(
                item,
                image,
                events,
                confounds,
                times,
                label,
                int(label.removeprefix("run-")),
            )
        )
    runs.sort(key=lambda run: run.number)
    if len({r.number for r in runs}) != len(runs):
        raise ValueError("NSD run numbers must be unique")
    brain = runs[0].image.header.get_axis(1)
    if any(r.image.header.get_axis(1) != brain for r in runs):
        raise ValueError("All runs must have identical grayordinate BrainModel axes")
    return tuple(runs), brain


def _preflight(output, paths):
    for name in paths:
        path = output / name
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"Refusing to replace existing output: {path}")
        for parent in path.parents:
            if parent == output.parent:
                break
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise ValueError("Output parents must be directories without symlinks")


def _empty_result(runs, n_features):
    return dict(
        betas=[
            np.full((len(r.events), n_features), np.nan, dtype=np.float32) for r in runs
        ],
        maps=np.full((3, n_features), np.nan, dtype=np.float32),
        provenance=[],
    )


def _fit_blocks(runs, root, brain, models, block_size):
    results = {name: _empty_result(runs, len(brain)) for name in models}
    for start in range(0, len(brain), block_size):
        stop = min(start + block_size, len(brain))
        signals = [
            np.asarray(r.image.dataobj[:, start:stop], dtype=float) for r in runs
        ]
        data = from_arrays(
            signals,
            [r.events for r in runs],
            frame_times=[r.frame_times for r in runs],
            confounds=[r.confounds for r in runs],
            sources=[_sources(r, root, np.arange(start, stop)) for r in runs],
        )
        for name, alpha in models.items():
            fit = fit_single_trials(
                data, ridge_alpha=alpha, run_labels=[r.label for r in runs]
            )
            target = results[name]
            for output, values in zip(target["betas"], fit.run_betas, strict=True):
                output[:, start:stop] = values
            target["maps"][:, start:stop] = np.stack(
                [fit.full_r2, fit.nuisance_r2, fit.delta_r2]
            )
            target["provenance"].append(fit.provenance.to_dict())
            if start == 0:
                target.update(
                    trial_table=fit.trial_table,
                    designs=fit.design_matrices,
                    diagnostics=fit.diagnostics,
                )
        print(f"Fitted grayordinates {start}:{stop} / {len(brain)}", flush=True)
    return results


def _model_metadata(runs, root, alpha, result):
    return {
        "ridge_alpha": alpha,
        "HRF": "canonical SPM; oversampling 50",
        "Estimator": "OLS" if alpha == 0 else "fixed normalized ridge",
        "Penalty": "unit-L2 trial columns after nuisance projection; nuisance unpenalized",
        "BetaUnits": "native input signal units; no scaling across runs",
        "Trials": "one map per event row; original order; no merging by image ID",
        "RT": "diagnostic only; finite positive RT and finite beta, centered within run on the same mask",
        "RTPartitions": "numeric BIDS run parity; all/odd/even/per-run descriptive Pearson r; no p-values",
        "Selection": "up to five cortical grayordinates by absolute odd-run OLS r; ties by index; even-run plots",
        "R2": "in-sample: 1 - pooled SSE / pooled within-run SST; each run has its own intercept",
        "DeltaR2": "actual estimator full R2 minus nuisance OLS R2; raw difference",
        "Undefined": "constant signals: NaN beta/R2; RT r also NaN for fewer than 3 valid trials or zero variance",
        "Confounds": {
            k: MODEL[k] for k in ("high_pass", "motion", "acompcor", "nonsteady_state")
        },
        "Sources": [r.inputs.bold.relative_to(root).as_posix() for r in runs],
        "RunLabels": [r.label for r in runs],
        "DesignDiagnostics": result["diagnostics"],
        "SoftwareVersions": {
            name: version(name)
            for name in ("boldtailor", "nilearn", "nibabel", "numpy", "pandas")
        },
    }


def _build_artifacts(runs, root, brain, models, results, paths, checks):
    rt = [
        pd.to_numeric(
            r.events.get("response_time", pd.Series(np.nan, index=r.events.index)),
            errors="coerce",
        ).to_numpy(dtype=float)
        for r in runs
    ]
    numbers = [r.number for r in runs]
    diagnostics = {
        name: correlate_rt(result["betas"], rt, run_numbers=numbers)
        for name, result in results.items()
    }
    artifacts = []
    for name, result in results.items():
        artifacts.extend(
            single_trial_artifacts(
                runs,
                brain,
                result,
                diagnostics[name],
                paths[name],
                _model_metadata(runs, root, models[name], result),
            )
        )
    cortex = np.isin(
        brain.name, ["CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"]
    )
    vertices = select_vertices(diagnostics["OLS"]["odd"], cortex)
    artifacts.append(
        table_artifact(checks[0], selected_vertex_table(vertices, brain, diagnostics))
    )
    artifacts.append(
        scatter_artifact(
            {name: result["betas"] for name, result in results.items()},
            rt,
            numbers,
            vertices,
            checks[1],
        )
    )
    return artifacts


def run_single_trial_analysis(
    bids_root=BIDS_ROOT,
    fmriprep_root=None,
    output_root=None,
    *,
    subject="sub-07",
    session="ses-nsd10",
    ridge_alpha=None,
    block_size=4096,
):
    """Fit every run and publish one complete set without overwriting old results."""
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, Integral)
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer")
    if not re.fullmatch(r"sub-[A-Za-z0-9]+", subject) or not re.fullmatch(
        r"ses-[A-Za-z0-9]+", session
    ):
        raise ValueError("subject and session must be BIDS labels")
    models = {"OLS": 0.0}
    if ridge_alpha is not None:
        alpha = validate_alpha(ridge_alpha)
        if alpha <= 0:
            raise ValueError(
                "optional ridge_alpha must be positive; OLS is always fitted"
            )
        models["Ridge"] = alpha
    root = Path(bids_root).resolve()
    prep = (
        Path(fmriprep_root) if fmriprep_root else root / "derivatives/fmriprep-25.2.5"
    ).resolve()
    if not prep.is_relative_to(root):
        raise ValueError(
            "fmriprep_root must be inside bids_root for relative provenance"
        )
    output = (
        Path(output_root) if output_root else root / "derivatives/boldtailor"
    ).absolute()
    inputs = discover_runs(root, prep, subject=subject, session=session)
    runs, brain = _load_runs(inputs)
    paths = {name: model_paths(runs, subject, session, name) for name in models}
    checks = diagnostic_paths(subject, session)
    _preflight(
        output,
        [p for model in paths.values() for p in all_model_paths(model)] + list(checks),
    )
    print(
        f"Loaded {len(runs)} runs, {sum(len(r.events) for r in runs)} trials; fixed settings {models}",
        flush=True,
    )
    results = _fit_blocks(runs, root, brain, models, block_size)
    artifacts = _build_artifacts(runs, root, brain, models, results, paths, checks)
    if not (output / "dataset_description.json").exists():
        artifacts.append(
            json_artifact(
                "dataset_description.json",
                {
                    "Name": "NSD single-trial models",
                    "BIDSVersion": "1.11.1",
                    "DatasetType": "derivative",
                    "GeneratedBy": [
                        {"Name": "boldtailor", "Version": version("boldtailor")}
                    ],
                },
            )
        )
    published = publish_artifact_set(
        output, artifacts, source_paths=[p for r in inputs for p in _input_paths(r)]
    )
    print(f"Saved {len(published)} files to {output}", flush=True)
    return published


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=BIDS_ROOT)
    parser.add_argument("--fmriprep-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--subject", default="sub-07")
    parser.add_argument("--session", default="ses-nsd10")
    parser.add_argument("--ridge-alpha", type=float)
    parser.add_argument("--block-size", type=int, default=4096)
    run_single_trial_analysis(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
