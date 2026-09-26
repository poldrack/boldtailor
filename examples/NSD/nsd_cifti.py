"""Fit one NSD session with boldtailor's public prepared-design API.

Run from this directory with ``uv run python nsd_cifti.py``.
CIFTI I/O and event transformations belong to this example, not boldtailor.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn.glm.first_level import compute_regressor

from boldtailor.prepared import PreparedDesignAnalysis
from boldtailor.prepared_fit import fit_prepared, task_delta_r2_prepared
from boldtailor.provenance import RunSources, SourceRef
from boldtailor.publication import Artifact, publish_artifact_set

BIDS_ROOT = Path("/Volumes/extdata1/NSD/BIDS")
MOTION = tuple(
    axis + suffix
    for axis in ("trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z")
    for suffix in ("", "_derivative1", "_power2", "_derivative1_power2")
)
CONTRASTS = {"stimulus": {"stimulus": 1.0}, "response_time": {"response_time": 1.0}}
MODEL = {
    "hrf": "spm",
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
class RunInputs:
    stem: str
    bold: Path
    events: Path
    confounds: Path
    confounds_json: Path
    bold_json: Path


@dataclass(frozen=True)
class PreparedRun:
    inputs: RunInputs
    image: nib.Cifti2Image
    design: pd.DataFrame
    frame_times: np.ndarray


def discover_runs(bids_root, fmriprep_root, *, subject="sub-07", session="ses-nsd10"):
    """Require a unique fsLR 91k CIFTI and complete metadata for every raw run."""
    raw = Path(bids_root) / subject / session / "func"
    derivative = Path(fmriprep_root) / subject / session / "func"
    events = sorted(raw.glob(f"{subject}_{session}_task-nsdcore_run-*_events.tsv"))
    if not events:
        raise FileNotFoundError(f"No NSD events found in {raw}")
    runs = []
    for event in events:
        stem = event.name.removesuffix("_events.tsv")
        bold = derivative / f"{stem}_space-fsLR_den-91k_bold.dtseries.nii"
        item = RunInputs(
            stem,
            bold,
            event,
            derivative / f"{stem}_desc-confounds_timeseries.tsv",
            derivative / f"{stem}_desc-confounds_timeseries.json",
            bold.with_name(bold.name.replace(".dtseries.nii", ".json")),
        )
        for path in _input_paths(item):
            if not path.is_file():
                raise FileNotFoundError(f"Missing input for {stem}: {path}")
        runs.append(item)
    found = set(
        derivative.glob(
            f"{subject}_{session}_task-nsdcore_run-*_space-fsLR_den-91k_bold.dtseries.nii"
        )
    )
    if found != {run.bold for run in runs}:
        raise ValueError("CIFTI runs and events runs do not match")
    return tuple(runs)


def _input_paths(run):
    return run.bold, run.events, run.confounds, run.confounds_json, run.bold_json


def select_confounds(table: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Select motion24, top six combined-mask aCompCor, cosines, and NSS spikes."""
    components = [
        name
        for name, info in metadata.items()
        if name.startswith("a_comp_cor_")
        and info.get("Retained") is True
        and info.get("Mask") == "combined"
        and name in table
    ]
    components.sort(
        key=lambda name: (-float(metadata[name]["VarianceExplained"]), name)
    )
    if len(components) < 6:
        raise ValueError("Require six retained combined-mask aCompCor components")
    cosine = sorted(name for name in table if name.startswith("cosine"))
    if not cosine:
        raise ValueError("Missing fMRIPrep cosine high-pass regressors")
    spikes = sorted(
        name for name in table if name.startswith("non_steady_state_outlier")
    )
    names = list(MOTION) + components[:6] + cosine + spikes
    missing = set(names) - set(table)
    if missing:
        raise ValueError(f"Missing confounds: {sorted(missing)}")
    selected = table.loc[:, names].apply(pd.to_numeric, errors="raise").copy()
    if selected.empty:
        raise ValueError("Confounds must have rows")
    # Only temporal derivatives are undefined at the first acquired volume.
    for name in MOTION:
        if "derivative1" in name and pd.isna(selected.iloc[0][name]):
            selected.loc[selected.index[0], name] = 0.0
    if not np.isfinite(selected.to_numpy()).all():
        raise ValueError("Selected confounds must be finite beyond initial derivatives")
    return selected


def task_regressors(events: pd.DataFrame, frame_times: np.ndarray) -> pd.DataFrame:
    """Convolve presentation and centered RT amplitudes separately with the SPM HRF."""
    values = events.loc[:, ["onset", "duration", "response_time"]].apply(pd.to_numeric)
    if values.empty or not np.isfinite(values.to_numpy()).all():
        raise ValueError(
            "onset, duration, and response_time must be present and finite"
        )
    if (values.duration <= 0).any() or (values.response_time <= 0).any():
        raise ValueError("duration and response_time must be positive")
    rt = values.response_time.to_numpy()
    amplitudes = {"stimulus": np.ones(len(values)), "response_time": rt - rt.mean()}
    columns = {}
    for name, amplitude in amplitudes.items():
        condition = np.vstack([values.onset, values.duration, amplitude])
        regressor, _ = compute_regressor(
            condition,
            "spm",
            frame_times,
            con_id=name,
            oversampling=50,
        )
        columns[name] = regressor[:, 0]
    return pd.DataFrame(columns)


def load_inputs(inputs: RunInputs):
    """Load image, raw events, selected nuisances, and corrected frame times."""
    image = nib.load(inputs.bold)
    if not isinstance(image, nib.Cifti2Image):
        raise ValueError(f"{inputs.stem}: expected CIFTI image")
    series, brain = (image.header.get_axis(i) for i in (0, 1))
    if not isinstance(series, nib.cifti2.SeriesAxis) or series.unit != "SECOND":
        raise ValueError("CIFTI time axis must be a SeriesAxis in seconds")
    if not isinstance(brain, nib.cifti2.BrainModelAxis):
        raise ValueError("CIFTI spatial axis must be a BrainModelAxis")
    timing = json.loads(inputs.bold_json.read_text())
    if not np.isclose(timing["RepetitionTime"], series.step):
        raise ValueError(f"{inputs.stem}: inconsistent repetition time")
    if timing.get("SliceTimingCorrected") and "StartTime" not in timing:
        raise ValueError(f"{inputs.stem}: slice-timing correction needs StartTime")
    offset = float(timing.get("StartTime", series.start))
    times = offset + np.arange(image.shape[0]) * series.step
    confounds = select_confounds(
        pd.read_csv(inputs.confounds, sep="\t"),
        json.loads(inputs.confounds_json.read_text()),
    )
    if len(confounds) != len(times):
        raise ValueError(f"{inputs.stem}: confound rows must match CIFTI volumes")
    return image, pd.read_csv(inputs.events, sep="\t"), confounds, times


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


def _source_ref(path, root, role, **annotations):
    stat = path.stat()
    return SourceRef(
        role=role,
        uri=path.relative_to(root).as_posix(),
        byte_size=stat.st_size,
        modified_at=datetime.fromtimestamp(stat.st_mtime, timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        annotations=annotations,
    )


def _sources(run, root, indices):
    inputs = run.inputs
    return RunSources(
        signal=_source_ref(
            inputs.bold,
            root,
            "signal",
            grayordinate_indices=indices.tolist(),
            sidecar=_source_ref(inputs.bold_json, root, "signal").to_dict(),
        ),
        events=_source_ref(inputs.events, root, "events"),
        confounds=_source_ref(
            inputs.confounds,
            root,
            "confounds",
            sidecar=_source_ref(inputs.confounds_json, root, "confounds").to_dict(),
        ),
    )


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
        sources=[_sources(run, root, indices) for run in runs],
        run_metadata=[{"run": run.inputs.stem} for run in runs],
    )
    full = fit_prepared(
        prepared, contrasts=CONTRASTS, noise_model="ols", model_metadata=MODEL
    )
    comparison = task_delta_r2_prepared(
        prepared,
        full,
        contrasts=CONTRASTS,
        noise_model="ols",
        model_metadata=MODEL,
    )
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


def _json_artifact(path, value):
    return Artifact(
        path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    )


def _scalar_artifacts(stem, brain, maps, metadata):
    names = [("full", "rsquared"), ("confounds", "rsquared"), ("task", "deltarsquared")]
    artifacts = []
    for (label, statistic), values in zip(names, maps, strict=True):
        name = f"{stem}_space-fsLR_den-91k_desc-{label}_stat-{statistic}"
        axes = (nib.cifti2.ScalarAxis([f"{label}_{statistic}"]), brain)
        image = nib.Cifti2Image(
            values[None].astype(np.float32), nib.Cifti2Header.from_axes(axes)
        )
        image.nifti_header.set_intent("ConnDenseScalar")
        artifacts.append(Artifact(f"{name}.dscalar.nii", image.to_bytes()))
        artifacts.append(_json_artifact(f"{name}.json", {**metadata, "Map": label}))
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
    artifacts.append(_json_artifact(f"{stem}_desc-model_metadata.json", metadata))
    artifacts.append(_json_artifact(f"{stem}_desc-blocks_provenance.json", provenance))
    return artifacts


def run_analysis(
    bids_root=BIDS_ROOT,
    fmriprep_root=None,
    output_root=None,
    *,
    subject="sub-07",
    session="ses-nsd10",
    block_size=4096,
):
    """Fit all session runs and atomically publish pooled maps and supporting files."""
    root = Path(bids_root).resolve()
    prep = (
        Path(fmriprep_root) if fmriprep_root else root / "derivatives/fmriprep-25.2.5"
    ).resolve()
    if not prep.is_relative_to(root):
        raise ValueError(
            "fmriprep_root must be inside bids_root for relative provenance"
        )
    output = Path(output_root) if output_root else root / "derivatives/boldtailor"
    inputs = discover_runs(root, prep, subject=subject, session=session)
    runs = tuple(_load_run(item) for item in inputs)
    print(
        f"Loaded designs for {len(runs)} runs; columns: {[len(run.design.columns) for run in runs]}",
        flush=True,
    )
    maps, provenance = _fit_session(runs, root, block_size)
    artifacts = _result_artifacts(runs, maps, provenance, root, subject, session)
    if not (output / "dataset_description.json").exists():
        artifacts.append(
            _json_artifact(
                "dataset_description.json",
                {
                    "Name": "NSD stimulus and response-time GLM",
                    "BIDSVersion": "1.11.1",
                    "DatasetType": "derivative",
                    "GeneratedBy": [
                        {"Name": "boldtailor", "Version": version("boldtailor")}
                    ],
                },
            )
        )
    paths = publish_artifact_set(
        output,
        artifacts,
        source_paths=[path for item in inputs for path in _input_paths(item)],
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
