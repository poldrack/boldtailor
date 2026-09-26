"""Expanded HRF NSD workflow, with odd-run training and even-run evaluation."""

from hashlib import sha256
from importlib.metadata import version

import nibabel as nib
import numpy as np
import pandas as pd

from boldtailor.data import from_arrays
from boldtailor import hrf_library, hrf_selection
from boldtailor._hrf_cv import prepare_runs
from boldtailor.single_trial import fit_selected_hrfs, fit_single_trials
from boldtailor.publication import publish_artifact_set

if __package__:
    from .nsd_single_trial import (
        _empty_result,
        _model_metadata,
        _preflight,
        _merge_block_arrays,
    )
    from .parallel_blocks import map_blocks, validate_n_jobs, execution_settings
    from .nsd_cifti import _sources, _input_paths
    from .rt_diagnostics import correlate_rt, select_vertices
    from .single_trial_artifacts import (
        all_model_paths,
        model_paths,
        json_artifact,
        single_trial_artifacts,
    )
    from .hrf_artifacts import (
        selection_paths,
        selection_artifacts,
        rt_artifacts,
        comparison_paths,
        comparison_artifacts,
    )
else:
    from nsd_single_trial import (
        _empty_result,
        _model_metadata,
        _preflight,
        _merge_block_arrays,
    )
    from parallel_blocks import map_blocks, validate_n_jobs, execution_settings
    from nsd_cifti import _sources, _input_paths
    from rt_diagnostics import correlate_rt, select_vertices
    from single_trial_artifacts import (
        all_model_paths,
        model_paths,
        json_artifact,
        single_trial_artifacts,
    )
    from hrf_artifacts import (
        selection_paths,
        selection_artifacts,
        rt_artifacts,
        comparison_paths,
        comparison_artifacts,
    )

SELECTION_STATS = (
    "hrfindex",
    "selectioncvr2",
    "canonicalcvr2",
    "deltacvr2",
    "oddhrfindex",
    "testr2",
    "canonicaltestr2",
    "deltatestr2",
)


def spatial_signature(brain, indices):
    header = nib.Cifti2Header.from_axes(
        (nib.cifti2.ScalarAxis(["spatial_identity"]), brain)
    )
    return sha256(
        header.to_xml() + np.asarray(indices, dtype="<i8").tobytes()
    ).hexdigest()


def block_data(runs, root, indices):
    indices = np.asarray(indices, dtype=int)
    # CIFTI proxies do not support arbitrary advanced indexing.
    start, stop = int(indices.min()), int(indices.max()) + 1
    return from_arrays(
        [
            np.asarray(r.image.dataobj[:, start:stop], dtype=float)[:, indices - start]
            for r in runs
        ],
        [r.events for r in runs],
        frame_times=[r.frame_times for r in runs],
        confounds=[r.confounds for r in runs],
        sources=[_sources(r, root, indices) for r in runs],
    )


def reaction_times(runs):
    return [
        pd.to_numeric(
            r.events.get("response_time", pd.Series(np.nan, index=r.events.index)),
            errors="coerce",
        ).to_numpy(dtype=float)
        for r in runs
    ]


def _collect_fit(target, fit, start, stop):
    for values, beta in zip(target["betas"], fit.run_betas, strict=True):
        values[:, start:stop] = beta
    target["maps"][:, start:stop] = np.stack(
        [fit.full_r2, fit.nuisance_r2, fit.delta_r2]
    )
    target["provenance"].append(fit.provenance.to_dict())
    target["trial_table"] = fit.trial_table
    for diagnostic in fit.diagnostics:
        key = (diagnostic["run_index"], diagnostic["hrf_id"])
        if key not in target["diagnostics"]:
            target["diagnostics"][key] = diagnostic.copy()
        else:
            target["diagnostics"][key]["n_features"] += diagnostic["n_features"]


def _store_selection(state, selection, evaluation, start, stop):
    ids = selection.hrf_indices.astype(float)
    ids[ids < 0] = np.nan
    values = [ids, selection.cv_r2, selection.canonical_cv_r2, selection.delta_cv_r2]
    if evaluation is not None:
        odd = evaluation.training_selection.hrf_indices.astype(float)
        odd[odd < 0] = np.nan
        values.extend(
            [
                odd,
                evaluation.test_r2,
                evaluation.canonical_test_r2,
                evaluation.delta_test_r2,
            ]
        )
        state["evaluation_provenance"].append(evaluation.provenance.to_dict())
    for name, value in zip(SELECTION_STATS, values, strict=False):
        state["maps"][name][start:stop] = value
    state["provenance"].append(selection.provenance.to_dict())
    tables = [("all", selection)]
    if evaluation is not None:
        tables.append(("odd", evaluation.training_selection))
    for scope, selected in tables:
        state["eligibility"].append(
            selected.eligibility.assign(
                scope=scope, feature_start=start, feature_stop=stop
            )
        )


def _canonical_rt(runs, root, indices, library, state):
    data = block_data(runs, root, indices)
    for run, design in zip(runs, prepare_runs(data, library), strict=True):
        eligible, reason = design.eligible(0)
        if not eligible:
            state["canonical_rt_reason"] = (
                f"Canonical HRF is ineligible in {run.label}: {reason}"
            )
            return np.full(len(indices), np.nan)
    canonical = fit_single_trials(data, run_labels=[r.label for r in runs])
    return correlate_rt(
        canonical.run_betas,
        reaction_times(runs),
        run_numbers=[r.number for r in runs],
    )["odd"]


def _canonical_comparisons(data, library, labels, models, results, start, stop):
    for label, design in zip(labels, prepare_runs(data, library), strict=True):
        eligible, reason = design.eligible(0)
        if not eligible:
            for result in results.values():
                result["canonical_reason"] = (
                    f"Canonical HRF is ineligible in {label}: {reason}"
                )
            return
    for name, alpha in models.items():
        fit = fit_single_trials(data, ridge_alpha=alpha, run_labels=labels)
        results[name]["canonical_r2"][start:stop] = fit.full_r2
        results[name]["canonical_provenance"].append(fit.provenance.to_dict())


def _empty_expanded(runs, n_features, models):
    results = {name: _empty_result(runs, n_features) for name in models}
    for result in results.values():
        result["diagnostics"] = {}
        result["canonical_r2"] = np.full(n_features, np.nan, dtype=np.float32)
        result["canonical_provenance"] = []
        result["canonical_reason"] = ""
    state = dict(
        maps={name: np.full(n_features, np.nan) for name in SELECTION_STATS},
        canonical_odd_rt=np.full(n_features, np.nan),
        provenance=[],
        evaluation_provenance=[],
        eligibility=[],
    )
    return results, state


def _fit_expanded_block(bounds, runs, root, brain, models, library, train, test):
    start, stop = bounds
    width = stop - start
    results, state = _empty_expanded(runs, width, models)
    available = len(train) >= 2 and len(test) >= 1
    labels = [r.label for r in runs]
    odd_runs = tuple(runs[i] for i in train)
    indices = np.arange(start, stop)
    signature = spatial_signature(brain, indices)
    data = block_data(runs, root, indices)
    selection = hrf_selection.select_hrf(
        data, library=library, run_labels=labels, feature_signature=signature
    )
    evaluation = (
        hrf_selection.evaluate_hrf_split(
            data,
            library=library,
            train_runs=train,
            test_runs=test,
            run_labels=labels,
            feature_signature=signature,
        )
        if available
        else None
    )
    _store_selection(state, selection, evaluation, 0, width)
    for name, alpha in models.items():
        fit = fit_selected_hrfs(
            data,
            selection=selection,
            ridge_alpha=alpha,
            run_labels=labels,
            feature_signature=signature,
        )
        _collect_fit(results[name], fit, 0, width)
    _canonical_comparisons(data, library, labels, models, results, 0, width)
    if odd_runs:
        state["canonical_odd_rt"][:] = _canonical_rt(
            odd_runs, root, indices, library, state
        )
    return results, state


def _merge_expanded_block(results, state, block, start, stop):
    fitted, selected = block
    for name, values in fitted.items():
        target = results[name]
        _merge_block_arrays(target, values, start, stop)
        target["canonical_r2"][start:stop] = values["canonical_r2"]
        target["canonical_provenance"].extend(values["canonical_provenance"])
        target["canonical_reason"] = values["canonical_reason"]
        for key, diagnostic in values["diagnostics"].items():
            if key not in target["diagnostics"]:
                target["diagnostics"][key] = diagnostic.copy()
            else:
                target["diagnostics"][key]["n_features"] += diagnostic["n_features"]
    for name, values in selected["maps"].items():
        state["maps"][name][start:stop] = values
    state["canonical_odd_rt"][start:stop] = selected["canonical_odd_rt"]
    for key in ("provenance", "evaluation_provenance"):
        state[key].extend(selected[key])
    state["eligibility"].extend(
        t.assign(feature_start=start, feature_stop=stop)
        for t in selected["eligibility"]
    )
    if "canonical_rt_reason" in selected:
        state["canonical_rt_reason"] = selected["canonical_rt_reason"]


def fit_expanded_blocks(
    runs, root, brain, models, block_size, library, train, test, n_jobs=1
):
    results, state = _empty_expanded(runs, len(brain), models)
    blocks = (
        (start, min(start + block_size, len(brain)))
        for start in range(0, len(brain), block_size)
    )
    args = (runs, root, brain, models, library, train, test)
    for (start, stop), block in map_blocks(
        _fit_expanded_block, blocks, args=args, n_jobs=n_jobs
    ):
        _merge_expanded_block(results, state, block, start, stop)
        print(
            f"Selected HRFs and fitted grayordinates {start}:{stop} / {len(brain)}",
            flush=True,
        )
    for result in results.values():
        result["diagnostics"] = list(result["diagnostics"].values())
        result["execution"] = execution_settings(n_jobs, block_size, len(brain))
    return results, state


def _independent_rt(runs, root, brain, models, library, state, train, test):
    cortex = np.isin(
        brain.name, ["CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"]
    )
    vertices = select_vertices(state["canonical_odd_rt"], cortex)
    if len(train) < 2 or not test or not len(vertices):
        return vertices, {}, None
    signature = spatial_signature(brain, vertices)
    data = block_data(runs, root, vertices)
    evaluation = hrf_selection.evaluate_hrf_split(
        data,
        library=library,
        train_runs=train,
        test_runs=test,
        run_labels=[r.label for r in runs],
        feature_signature=signature,
    )
    expected = state["maps"]["oddhrfindex"][vertices]
    actual = evaluation.training_selection.hrf_indices.astype(float)
    actual[actual < 0] = np.nan
    if not np.array_equal(expected, actual, equal_nan=True):
        raise RuntimeError(
            "selected-vertex HRFs differ from odd-run feature-block selection"
        )
    even_runs = tuple(runs[i] for i in test)
    even_data = block_data(even_runs, root, vertices)
    fits = {
        name: fit_selected_hrfs(
            even_data,
            selection=evaluation.training_selection,
            ridge_alpha=alpha,
            run_labels=[r.label for r in even_runs],
            feature_signature=signature,
        )
        for name, alpha in models.items()
    }
    return vertices, fits, evaluation


def _metadata(library, runs, train, test):
    available = len(train) >= 2 and bool(test)
    return dict(
        LibraryFingerprint=library.fingerprint,
        CandidateCount=len(library.candidates),
        HRFNormalization="discrete sum one; curves at 0.1 s; convolution at TR/50",
        CanonicalAnchor="candidate 0: exact Nilearn spm, 32-second support",
        TrainingRuns=[runs[i].label for i in train],
        TestRuns=[runs[i].label for i in test],
        IndependentEvaluationAvailable=available,
        IndependentEvaluationReason=(
            ""
            if available
            else "Requires at least two odd training runs and one even test run"
        ),
        SelectionScore="nuisance-adjusted mean-stimulus leave-one-run-out CV R2; used for HRF selection, not independent performance",
        TestScore="odd-selected HRF and odd-trained mean amplitude frozen for even runs; nuisance projected per run",
        RT="production maps descriptive; vertices selected using canonical OLS odd-run RT; even free trial betas use odd-selected HRFs",
        GroupedDesigns="one float64 NPZ per run: hrf_<id> trial matrices, shared nuisance, frame_times and string IDs; allow_pickle=False",
        Undefined="NaN HRF IDs/scores/betas for numerically zero signal outside nuisance span",
        FeatureIdentity="SHA256 of ordered BrainModel axis XML and grayordinate indices",
        SoftwareVersions={
            name: version(name) for name in ("boldtailor", "nilearn", "numpy", "scipy")
        },
    )


def run_expanded_analysis(
    runs, root, output, brain, models, block_size, subject, session, n_jobs=1
):
    n_jobs = validate_n_jobs(n_jobs)
    paths = {
        name: model_paths(runs, subject, session, name, descriptor="hrfOpt" + name)
        for name in models
    }
    for model_paths_ in paths.values():
        model_paths_.update(comparison_paths(model_paths_["fullrsquared"]))
    selection_paths_ = selection_paths(runs, subject, session)
    _preflight(
        output,
        [p for model in paths.values() for p in all_model_paths(model)]
        + list(all_model_paths(selection_paths_)),
    )
    library = hrf_library.expanded_hrf_library()
    train = [i for i, r in enumerate(runs) if r.number % 2 == 1]
    test = [i for i, r in enumerate(runs) if r.number % 2 == 0]
    print(
        f"Expanded HRF analysis: {len(runs)} runs, {len(library.candidates)} candidates, {len(brain)} grayordinates, {n_jobs} requested workers",
        flush=True,
    )
    results, state = fit_expanded_blocks(
        runs, root, brain, models, block_size, library, train, test, n_jobs=n_jobs
    )
    vertices, rt_fits, evaluation = _independent_rt(
        runs, root, brain, models, library, state, train, test
    )
    metadata = _metadata(library, runs, train, test)
    metadata["Execution"] = execution_settings(n_jobs, block_size, len(brain))
    if "canonical_rt_reason" in state:
        metadata.update(
            IndependentRTAvailable=False,
            IndependentRTReason=state["canonical_rt_reason"],
        )
    artifacts = list(
        selection_artifacts(
            runs, brain, library, state, selection_paths_, metadata, train, test
        )
    )
    for name, result in results.items():
        meta = _model_metadata(runs, root, models[name], result)
        meta.update(
            HRF="per-grayordinate all-run CV selection; sum-normalized",
            RT="descriptive production correlations; all-run selected HRFs",
            LibraryFingerprint=library.fingerprint,
            Selection="Independent diagnostic vertices use canonical odd-run OLS; see hrfSelection metadata",
            GroupedDesigns=selection_paths_["designs"],
        )
        diagnostics = correlate_rt(
            result["betas"], reaction_times(runs), run_numbers=[r.number for r in runs]
        )
        artifacts.extend(
            single_trial_artifacts(runs, brain, result, diagnostics, paths[name], meta)
        )
        artifacts.extend(
            comparison_artifacts(
                brain,
                result["maps"][0],
                result["canonical_r2"],
                paths[name],
                _comparison_metadata(meta, result, paths[name]),
            )
        )
    artifacts.extend(
        rt_artifacts(
            runs,
            brain,
            library,
            state,
            vertices,
            rt_fits,
            evaluation,
            selection_paths_,
            test,
        )
    )
    if not (output / "dataset_description.json").exists():
        artifacts.append(
            json_artifact(
                "dataset_description.json",
                dict(
                    Name="NSD expanded HRF single-trial models",
                    BIDSVersion="1.11.1",
                    DatasetType="derivative",
                    GeneratedBy=[
                        dict(Name="boldtailor", Version=version("boldtailor"))
                    ],
                ),
            )
        )
    published = publish_artifact_set(
        output,
        artifacts,
        source_paths=[p for r in runs for p in _input_paths(r.inputs)],
    )
    print(f"Saved {len(published)} files to {output}", flush=True)
    return published


def _comparison_metadata(model_metadata, result, paths):
    keys = (
        "ridge_alpha",
        "Estimator",
        "Penalty",
        "R2",
        "Confounds",
        "Sources",
        "RunLabels",
        "SoftwareVersions",
        "Execution",
    )
    return dict(
        {key: model_metadata[key] for key in keys},
        ComparisonAvailable=not bool(result["canonical_reason"]),
        ComparisonUnavailableReason=result["canonical_reason"],
        CanonicalFitProvenance=result["canonical_provenance"],
        OptimizedFitProvenance=paths["provenance"],
    )
