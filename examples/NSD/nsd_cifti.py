"""Fit one NSD session with boldtailor's public prepared-design API.

Run from this directory with ``uv run python nsd_cifti.py``.
CIFTI I/O and event transformations belong to this example, not boldtailor.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import compute_regressor

from boldtailor.design import event_response_scales, hrf_model
from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.prepared_fit import fit_prepared, task_delta_r2_prepared
from boldtailor.cifti import scalar_artifact
from boldtailor.publication import Artifact, publish_artifact_set

if __package__:
    from .notebook_paths import notebook_paths
    from .workflow_artifacts import dataset_description, json_artifact
    from .workflow_files import (
        RunInputs,
        discover_runs,
        input_paths,
        load_inputs,
        run_sources,
    )
else:
    from notebook_paths import notebook_paths
    from workflow_artifacts import dataset_description, json_artifact
    from workflow_files import (
        RunInputs,
        discover_runs,
        input_paths,
        load_inputs,
        run_sources,
    )

BIDS_ROOT = None  # Use --bids-root or NSD_BIDS_ROOT; no personal default.
CONTRASTS = {"stimulus": {"stimulus": 1.0}, "response_time": {"response_time": 1.0}}
MODEL = {
    "hrf": "spm",
    "hrf_normalization": "peak_one_event_response",
    "oversampling": 50,
    "response_time": "within-run mean-centered seconds; no orthogonalization",
    "duration": "recorded stimulus duration",
    "high_pass": "fMRIPrep cosine columns; no additional temporal filtering",
    "motion": "24 terms: six parameters, derivatives, and their squares",
    "acompcor": "six largest retained combined-mask components by explained variance",
    "nonsteady_state": "one indicator per flagged volume",
    "signal_scaling": "none",
    "noise_model": "ols",
}


@dataclass(frozen=True)
class PreparedRun:
    inputs: RunInputs
    image: nib.Cifti2Image
    design: pd.DataFrame
    frame_times: np.ndarray


def task_regressors(events: pd.DataFrame, frame_times: np.ndarray) -> pd.DataFrame:
    """Convolve presentation and centered RT with the SPM HRF, unit event peaks."""
    values = events.loc[:, ["onset", "duration", "response_time"]].apply(pd.to_numeric)
    if values.empty or not np.isfinite(values.to_numpy()).all():
        raise ValueError(
            "onset, duration, and response_time must be present and finite"
        )
    if (values.duration <= 0).any() or (values.response_time <= 0).any():
        raise ValueError("duration and response_time must be positive")
    rt = values.response_time.to_numpy()
    amplitudes = {"stimulus": np.ones(len(values)), "response_time": rt - rt.mean()}
    scales = event_response_scales(
        hrf_model("spm"), values.onset, values.duration, frame_times
    )
    columns = {}
    for name, amplitude in amplitudes.items():
        condition = np.vstack([values.onset, values.duration, amplitude * scales])
        regressor, _ = compute_regressor(
            condition,
            hrf_model("spm"),
            frame_times,
            con_id=name,
            oversampling=50,
        )
        columns[name] = regressor[:, 0]
    return pd.DataFrame(columns)


def _load_run(inputs: RunInputs) -> PreparedRun:
    image, events, confounds, times = load_inputs(inputs)
    task = task_regressors(events, times)
    design = pd.concat([task, confounds], axis=1).assign(constant=1.0)
    matrix = design.to_numpy()
    if (
        np.linalg.matrix_rank(matrix) != matrix.shape[1]
        or len(design) <= matrix.shape[1]
    ):
        raise ValueError(
            f"{inputs.stem}: design requires full rank and residual degrees of freedom"
        )
    return PreparedRun(inputs, image, design, times)


def _fit_block(runs, signals, root, indices):
    roles = [
        {
            name: (
                "task"
                if name in CONTRASTS
                else "intercept" if name == "constant" else "nuisance"
            )
            for name in run.design
        }
        for run in runs
    ]
    prepared = PreparedDesignAnalysis.from_arrays(
        signals=signals,
        design_matrices=[run.design for run in runs],
        frame_times=[run.frame_times for run in runs],
        column_roles=roles,
        sources=[run_sources(run, root, indices) for run in runs],
        run_metadata=[{"run": run.inputs.stem} for run in runs],
    )
    full = fit_prepared(
        prepared, contrasts=CONTRASTS, noise_model="ols", model_metadata=MODEL
    )
    comparison = task_delta_r2_prepared(prepared, full)
    maps = np.stack(
        [comparison.full_r2, comparison.nuisance_r2, comparison.raw_delta_r2]
    )
    return maps, comparison.provenance.to_dict()


def _fit_session(runs, root, block_size):
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    brain = runs[0].image.header.get_axis(1)
    if any(run.image.header.get_axis(1) != brain for run in runs[1:]):
        raise ValueError("All runs must have identical grayordinate BrainModel axes")
    maps = np.full((3, len(brain)), np.nan)
    provenance = []
    for start in range(0, len(brain), block_size):
        stop = min(start + block_size, len(brain))
        signals = [
            np.asarray(run.image.dataobj[:, start:stop], dtype=float) for run in runs
        ]
        if not all(np.isfinite(y).all() for y in signals):
            raise ValueError(f"Nonfinite CIFTI signals in grayordinates {start}:{stop}")
        total_ss = sum(np.sum((y - y.mean(axis=0)) ** 2, axis=0) for y in signals)
        indices = np.arange(start, stop)[total_ss > 0]
        if len(indices):
            values, record = _fit_block(
                runs, [y[:, total_ss > 0] for y in signals], root, indices
            )
            maps[:, indices] = values
            provenance.append(record)
        print(f"Fitted grayordinates {start}:{stop} / {len(brain)}", flush=True)
    return maps, provenance


def _scalar_artifacts(stem, brain, maps, metadata):
    names = [("full", "rsquared"), ("confounds", "rsquared"), ("task", "deltarsquared")]
    artifacts = []
    for (label, statistic), values in zip(names, maps, strict=True):
        name = f"{stem}_space-fsLR_den-91k_desc-{label}_stat-{statistic}"
        artifacts.append(
            scalar_artifact(
                f"{name}.dscalar.nii", brain, values[None], [f"{label}_{statistic}"]
            )
        )
        artifacts.append(json_artifact(f"{name}.json", {**metadata, "Map": label}))
    return artifacts


def _result_artifacts(runs, maps, provenance, root, subject, session):
    prefix = f"{subject}/{session}/func"
    stem = f"{prefix}/{subject}_{session}_task-nsdcore"
    metadata = {
        "Model": MODEL,
        "Runs": [run.inputs.stem for run in runs],
        "Sources": [run.inputs.bold.relative_to(root).as_posix() for run in runs],
        "R2Definition": "1 - sum_run(SSE_run) / sum_run(sum_time((y_run - mean_time(y_run))^2))",
        "DeltaR2Definition": "full R2 - confounds-only R2; raw difference, no clipping",
        "Fit": "in-sample OLS, independent coefficients and intercepts per run",
        "UndefinedR2": "NaN where total within-run sum of squares is zero",
        "UndefinedGrayordinates": int(np.isnan(maps[0]).sum()),
        "SoftwareVersions": {
            name: version(name)
            for name in ("boldtailor", "nilearn", "nibabel", "numpy", "pandas")
        },
    }
    artifacts = _scalar_artifacts(
        stem, runs[0].image.header.get_axis(1), maps, metadata
    )
    for run in runs:
        frame = run.design.copy()
        frame.insert(0, "frame_time", run.frame_times)
        artifacts.append(
            Artifact(
                f"{prefix}/{run.inputs.stem}_desc-full_design.tsv",
                frame.to_csv(sep="\t", index=False).encode(),
            )
        )
    artifacts.append(json_artifact(f"{stem}_desc-model_metadata.json", metadata))
    artifacts.append(json_artifact(f"{stem}_desc-blocks_provenance.json", provenance))
    return artifacts


def script_roots(bids_root=None, fmriprep_root=None, output_root=None):
    """Resolve roots like the notebooks: arguments, then NSD_* variables, then defaults."""
    given = dict(
        bids_root=bids_root, fmriprep_root=fmriprep_root, output_root=output_root
    )
    paths = notebook_paths({k: v for k, v in given.items() if v is not None})
    root = Path(paths["bids_root"]).resolve()
    prep = Path(paths["fmriprep_root"]).resolve()
    if not prep.is_relative_to(root):
        raise ValueError(
            "fmriprep_root must be inside bids_root for relative provenance"
        )
    return root, prep, Path(paths["output_root"])


def run_analysis(
    bids_root=BIDS_ROOT,
    fmriprep_root=None,
    output_root=None,
    *,
    subject="sub-07",
    session="ses-nsd10",
    block_size=4096,
):
    """Fit session runs and publish maps with per-file replacement and rollback."""
    root, prep, output = script_roots(bids_root, fmriprep_root, output_root)
    inputs = discover_runs(root, prep, subject=subject, session=session)
    runs = tuple(_load_run(item) for item in inputs)
    print(
        f"Loaded designs for {len(runs)} runs; columns: {[len(run.design.columns) for run in runs]}",
        flush=True,
    )
    maps, provenance = _fit_session(runs, root, block_size)
    artifacts = _result_artifacts(runs, maps, provenance, root, subject, session)
    if not (output / "dataset_description.json").exists():
        artifacts.append(dataset_description("NSD stimulus and response-time GLM"))
    paths = publish_artifact_set(
        output,
        artifacts,
        source_paths=[path for item in inputs for path in input_paths(item)],
    )
    for name, values in zip(("full R2", "confounds R2", "delta R2"), maps, strict=True):
        finite = values[np.isfinite(values)]
        print(
            f"{name}: median={np.median(finite):.6f}"
            if len(finite)
            else f"{name}: undefined"
        )
    print(f"Saved {len(paths)} files to {output}", flush=True)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bids-root", type=Path, default=BIDS_ROOT)
    parser.add_argument("--fmriprep-root", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--subject", default="sub-07")
    parser.add_argument("--session", default="ses-nsd10")
    parser.add_argument("--block-size", type=int, default=4096)
    run_analysis(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
