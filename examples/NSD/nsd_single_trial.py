"""Fit NSD trial betas with canonical or selected HRFs and optional fixed ridge.

Run ``uv run python examples/NSD/nsd_single_trial.py --ridge-alpha 0.1``
from the repository root. RT is used only for descriptive checks after fitting.
"""

import argparse
from importlib.metadata import version
from numbers import Integral
from pathlib import Path

import numpy as np

from boldtailor.data import from_arrays
from boldtailor.single_trial import fit_single_trials
from boldtailor.single_trial import validate_alpha
from boldtailor.diagnostics import correlate_rt
from boldtailor.parallel import map_blocks, validate_n_jobs
from boldtailor.publication import publish_artifact_set

if __package__:
    from .parallel_blocks import execution_settings
    from .nsd_cifti import (
        BIDS_ROOT,
        MODEL,
        discover_runs,
        script_roots,
    )
    from .rt_diagnostics import scatter_artifact, select_vertices
    from boldtailor.workflow.artifacts import dataset_description, table_artifact
    from boldtailor.workflow.files import (
        bids_label,
        input_paths,
        load_runs,
        run_sources,
    )
    from boldtailor.workflow.files import reaction_times
    from .single_trial_artifacts import (
        all_model_paths,
        diagnostic_paths,
        model_paths,
        selected_vertex_table,
        single_trial_artifacts,
    )
else:
    from parallel_blocks import execution_settings
    from nsd_cifti import (
        BIDS_ROOT,
        MODEL,
        discover_runs,
        script_roots,
    )
    from rt_diagnostics import scatter_artifact, select_vertices
    from boldtailor.workflow.artifacts import dataset_description, table_artifact
    from boldtailor.workflow.files import (
        bids_label,
        input_paths,
        load_runs,
        run_sources,
    )
    from boldtailor.workflow.files import reaction_times
    from single_trial_artifacts import (
        all_model_paths,
        diagnostic_paths,
        model_paths,
        selected_vertex_table,
        single_trial_artifacts,
    )


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


def _fit_trial_block(bounds, runs, root, models):
    start, stop = bounds
    data = from_arrays(
        [np.asarray(r.image.dataobj[:, start:stop], dtype=float) for r in runs],
        [r.events for r in runs],
        frame_times=[r.frame_times for r in runs],
        confounds=[r.confounds for r in runs],
        sources=[run_sources(r, root, np.arange(start, stop)) for r in runs],
    )
    results = {}
    for name, alpha in models.items():
        fit = fit_single_trials(
            data, ridge_alpha=alpha, run_labels=[r.label for r in runs]
        )
        results[name] = dict(
            betas=[np.asarray(b, dtype=np.float32) for b in fit.run_betas],
            maps=np.asarray(
                [fit.full_r2, fit.nuisance_r2, fit.delta_r2], dtype=np.float32
            ),
            provenance=[fit.provenance.to_dict()],
            trial_table=fit.trial_table,
            designs=fit.design.matrices,
            diagnostics=fit.diagnostics,
        )
    return results


def _merge_block_arrays(target, block, start, stop):
    for output, values in zip(target["betas"], block["betas"], strict=True):
        output[:, start:stop] = values
    target["maps"][:, start:stop] = block["maps"]
    target["provenance"].extend(block["provenance"])
    target["trial_table"] = block["trial_table"]


def _fit_blocks(runs, root, brain, models, block_size, n_jobs=1):
    results = {name: _empty_result(runs, len(brain)) for name in models}
    blocks = (
        (start, min(start + block_size, len(brain)))
        for start in range(0, len(brain), block_size)
    )
    for (start, stop), block in map_blocks(
        _fit_trial_block, blocks, args=(runs, root, models), n_jobs=n_jobs
    ):
        for name, fitted in block.items():
            _merge_block_arrays(results[name], fitted, start, stop)
            if start == 0:
                results[name].update(
                    designs=fitted["designs"], diagnostics=fitted["diagnostics"]
                )
        print(f"Fitted grayordinates {start}:{stop} / {len(brain)}", flush=True)
    for result in results.values():
        result["execution"] = execution_settings(n_jobs, block_size, len(brain))
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
        "Execution": result["execution"],
        "SoftwareVersions": {
            name: version(name)
            for name in ("boldtailor", "nilearn", "nibabel", "numpy", "pandas")
        },
    }


def _build_artifacts(runs, root, brain, models, results, paths, checks):
    rt = reaction_times(runs, missing_ok=True)
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
    hrf_library="canonical",
    n_jobs=1,
):
    """Fit every run and publish one complete set without overwriting old results."""
    n_jobs = validate_n_jobs(n_jobs)
    if hrf_library not in ("canonical", "expanded"):
        raise ValueError("hrf_library must be canonical or expanded")
    if (
        isinstance(block_size, bool)
        or not isinstance(block_size, Integral)
        or block_size <= 0
    ):
        raise ValueError("block_size must be a positive integer")
    if not bids_label(subject, "sub") or not bids_label(session, "ses"):
        raise ValueError("subject and session must be BIDS labels")
    models = {"OLS": 0.0}
    if ridge_alpha is not None:
        alpha = validate_alpha(ridge_alpha)
        if alpha <= 0:
            raise ValueError(
                "optional ridge_alpha must be positive; OLS is always fitted"
            )
        models["Ridge"] = alpha
    root, prep, output = script_roots(bids_root, fmriprep_root, output_root)
    output = output.absolute()
    inputs = discover_runs(root, prep, subject=subject, session=session)
    runs, brain = load_runs(inputs)
    if hrf_library == "expanded":
        if __package__:
            from .nsd_hrf import run_expanded_analysis
        else:
            from nsd_hrf import run_expanded_analysis
        return run_expanded_analysis(
            runs,
            root,
            output,
            brain,
            models,
            block_size,
            subject,
            session,
            n_jobs=n_jobs,
        )
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
    results = _fit_blocks(runs, root, brain, models, block_size, n_jobs=n_jobs)
    artifacts = _build_artifacts(runs, root, brain, models, results, paths, checks)
    if not (output / "dataset_description.json").exists():
        artifacts.append(dataset_description("NSD single-trial models"))
    published = publish_artifact_set(
        output, artifacts, source_paths=[p for r in inputs for p in input_paths(r)]
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
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel feature-block processes (default: 1)",
    )
    parser.add_argument(
        "--hrf-library", choices=("canonical", "expanded"), default="canonical"
    )
    run_single_trial_analysis(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
