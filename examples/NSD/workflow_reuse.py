"""Reload published numerical results for notebook inspection without refitting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from boldtailor.cifti import read_scalar
from boldtailor.model import HRF_NORMALIZATION
from boldtailor.workflow.inputs import NSD_TASK_MODEL
from .workflow_outputs import _stem

ANALYSIS_SETTINGS = (
    "bids_root",
    "fmriprep_root",
    "subject",
    "session",
    "max_grayordinates",
    "encoding_mode",
    "ridge_mode",
    "ridge_fractions",
    "ridge_alphas",
    "ridge_percentile",
    "ridge_alpha",
    "hrf_library",
    "hrf_selection_rt",
    "hrf_n_samples",
    "hrf_seed",
    "hrf_parameters",
)


def validate_saved_settings(metadata, settings):
    """Execution and display settings may change; analysis settings must agree."""
    saved = metadata["settings"]
    changed = [key for key in ANALYSIS_SETTINGS if saved.get(key) != settings.get(key)]
    if changed:
        raise ValueError(
            "Saved analysis settings differ: "
            + ", ".join(changed)
            + "; use existing_results='overwrite' to refit"
        )
    identities = {
        "task_model_fingerprint": NSD_TASK_MODEL.fingerprint,
        "hrf_normalization": HRF_NORMALIZATION,
    }
    for key, expected in identities.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"Saved analysis settings differ: {key}; "
                "use existing_results='overwrite' to refit"
            )


def read_json(path):
    return json.loads(Path(path).read_text())


def read_map(base, descriptor, statistic, brain):
    path = f"{base}_space-fsLR_den-91k_desc-notebook{descriptor}_stat-{statistic}.dscalar.nii"
    return read_scalar(path, brain)


def _designs(base, descriptor, runs):
    result = {}
    with np.load(f"{base}_desc-notebook{descriptor}_designs.npz") as archive:
        for i, run in enumerate(runs):
            prefix = f"{run.label}_hrf-"
            for key in archive.files:
                if key.startswith(prefix) and not key.endswith(
                    ("_columns", "_frame_times")
                ):
                    if not np.array_equal(
                        archive[key + "_frame_times"], run.frame_times
                    ):
                        raise ValueError(f"Saved frame times differ for {run.label}")
                    result[i, int(key[len(prefix) :])] = pd.DataFrame(
                        archive[key],
                        columns=archive[key + "_columns"],
                        index=run.frame_times,
                    )
    return result


def _glm(base, descriptor, runs, brain):
    result = {
        key: read_map(base, descriptor, key, brain)
        for key in ("effects", "variances", "t", "z")
    }
    result.update(
        r2=read_map(base, descriptor, "rsquared", brain),
        designs=_designs(base, descriptor, runs),
    )
    return result


def _beta(base, descriptor, runs, brain):
    table = pd.read_csv(f"{base}_desc-notebook{descriptor}_trials.tsv", sep="\t")
    betas = [
        read_scalar(
            base.parent
            / f"{run.inputs.stem}_space-fsLR_den-91k_desc-notebook{descriptor}_betas.dscalar.nii",
            brain,
        )
        for run in runs
    ]
    for run, values in zip(runs, betas, strict=True):
        if len(values) != len(run.events):
            raise ValueError(f"Saved trial count differs for {run.label}")
    rt = read_map(base, descriptor, "rtcorrelation", brain)
    return dict(
        betas=betas,
        trial_table=table,
        r2=read_map(base, descriptor, "rsquared", brain),
        rt=dict(zip(("all", "odd", "even"), rt, strict=True)),
    )


def _beta_names(settings):
    names = ["CanonicalTrialOLS", "OptimizedTrialOLS"]
    suffix = {"fixed": "Ridge", "cv": "RidgeCV", "fractional_cv": "FractionalCV"}
    if settings["ridge_mode"] in suffix:
        names.extend(
            h + "Trial" + suffix[settings["ridge_mode"]]
            for h in ("Canonical", "Optimized")
        )
    return names


def _hrf_maps(base, brain):
    result = {
        scope: read_map(base, "HRF" + scope.title(), "selection", brain)
        for scope in ("all", "odd", "even")
    }
    for scope in ("odd_to_even", "even_to_odd"):
        descriptor = "HRF" + "".join(word.title() for word in scope.split("_"))
        result[scope] = read_map(base, descriptor, "prediction", brain)
    return result


def load_saved_workflow(output, runs, settings):
    """Load the plotting inputs from existing CIFTI/TSV/NPZ/JSON exports.

    Reuse reads the saved analysis, not a content-addressed cache of current raw
    data. Settings, run lengths, and spatial axes are checked. Refit explicitly
    after changing input file contents. No pickle or fitting objects are loaded.
    """
    base = Path(output) / _stem(settings["subject"], settings["session"])
    try:
        metadata = read_json(f"{base}_desc-notebook_metadata.json")
        validate_saved_settings(metadata, settings)
        if metadata["retained_scans"] != [len(r.frame_times) for r in runs]:
            raise ValueError("Saved scan counts differ from current inputs")
        brain = runs[0].image.header.get_axis(1)
        result = dict(
            metadata=metadata,
            glms={
                name: _glm(base, name, runs, brain)
                for name in ("CanonicalGLM", "OptimizedGLM")
            },
            hrf_maps=_hrf_maps(base, brain),
            beta_models={
                name: _beta(base, name, runs, brain) for name in _beta_names(settings)
            },
            ridge_cv={},
        )
        if settings["ridge_mode"] in ("cv", "fractional_cv"):
            from .ridge_reuse import load_ridge_summaries

            result["ridge_cv"] = load_ridge_summaries(base, brain, settings)
        return result
    except FileNotFoundError as error:
        raise ValueError(
            f"Saved notebook results are incomplete: {error}; use overwrite to refit"
        ) from error
