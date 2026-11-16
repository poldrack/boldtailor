"""Fixed-reference mean models and streamed HRF random-effect designs."""

from hashlib import sha256

import numpy as np
import pandas as pd

from boldtailor._hrf_cv import prepare_runs
from boldtailor._single_trial_design import compile_trial_run
from boldtailor.reml import RemlGroup, fit_reml, select_random_effects_design
from boldtailor.hrf_results import RemlHrfDiagnostics
from boldtailor._hrf_design import HRF_NORMALIZATION, MIN_ONSET, OVERSAMPLING


def prepare_reml_groups(data, library, task_model, options, labels):
    """Build one fixed mean/nuisance span per run, shared by all candidates."""
    options.validate(data.n_features, library)
    runs = prepare_runs(data, library, task_model, labels=labels)
    return tuple(
        RemlGroup(
            y,
            np.column_stack(
                [run.nuisance, _reference_task_columns(run, options.reference_hrf_id)]
            ),
        )
        for y, run in zip(data.signals, runs, strict=True)
    )


def iter_trial_designs(data, library, candidate_ids, labels):
    """Yield one candidate at a time without imposing trial OLS rank rules."""
    inputs = tuple(
        zip(data.events, data.frame_times, data.confounds, labels, strict=True)
    )
    for cid in candidate_ids:
        yield tuple(
            compile_trial_run(
                events, times, confounds, label, hrf=library.candidates[cid]
            )[0].to_numpy()
            for events, times, confounds, label in inputs
        )


def _reference_task_columns(run, reference_id):
    """Aggregate unit-peak trial columns with the shared task-model amplitudes."""
    expanded = run.expanded
    weights = np.column_stack(
        [
            expanded.loc[expanded.trial_type == name, "modulation"].to_numpy()
            for name in dict.fromkeys(expanded.trial_type)
        ]
    )
    return run.trial_matrix(reference_id) @ weights


def _array_fingerprint(values):
    array = np.asarray(values, dtype="<f8")
    return sha256(str(array.shape).encode() + array.tobytes()).hexdigest()


def _reml_activity(
    groups, library, labels, signature, task_model, options, ids, selected
):
    """Describe the fixed statistical model independently of execution settings."""
    fixed = [_array_fingerprint(g.fixed_design) for g in groups]
    return dict(
        name="hrf_selection",
        method="reml",
        selection_statistic=True,
        library=dict(library.origin),
        library_fingerprint=library.fingerprint,
        design_fingerprint=sha256("|".join(fixed).encode()).hexdigest(),
        fixed_design_fingerprints=fixed,
        hrf_assignment_fingerprint=sha256(ids.astype("<i8").tobytes()).hexdigest(),
        run_labels=list(labels),
        feature_signature=signature,
        task_model=task_model.to_dict(),
        task_model_fingerprint=task_model.fingerprint,
        task_regressors=list(task_model.regressor_names),
        profiled_regressors=list(task_model.profiled_names),
        reference_hrf_id=int(options.reference_hrf_id),
        rho=np.asarray(options.rho).tolist(),
        score="negative_twice_restricted_log_likelihood",
        fixed_mean="reference_hrf_task_model_and_run_nuisance",
        random_effects="independent_zero_mean_trial_coefficients",
        variance_pooling="per_feature_shared_across_runs",
        hrf_normalization=HRF_NORMALIZATION,
        beta_units="native_signal",
        min_onset=MIN_ONSET,
        oversampling=OVERSAMPLING,
        provisional_features=int(np.count_nonzero(selected.any_unrefined)),
        unsupported_features=int(np.count_nonzero(ids < 0)),
    )


def _diagnostics(data, library, groups, options, labels, candidate_ids, selected):
    canonical_design = next(iter_trial_designs(data, library, [0], labels))
    canonical = fit_reml(groups, canonical_design, rho=options.rho)
    delta = np.full(data.n_features, np.nan)
    valid = (selected.winner >= 0) & np.isfinite(canonical.criterion)
    delta[valid] = canonical.criterion[valid] - selected.fit.criterion[valid]
    return RemlHrfDiagnostics(
        selection=selected,
        candidate_ids=candidate_ids,
        canonical_fit=canonical,
        delta_criterion=delta,
        reference_hrf_id=options.reference_hrf_id,
    )


def select_reml_hrfs(data, library, labels, signature, task_model, options):
    """Adapt covariance selection to an identified spatial HRF assignment."""
    groups = prepare_reml_groups(data, library, task_model, options, labels)
    candidate_ids = np.arange(len(library.candidates))
    candidates = iter_trial_designs(data, library, candidate_ids, labels)
    selected = select_random_effects_design(groups, candidates, rho=options.rho)
    ids = np.full(data.n_features, -1, dtype=np.int64)
    supported = selected.winner >= 0
    ids[supported] = candidate_ids[selected.winner[supported]]
    diagnostics = _diagnostics(
        data, library, groups, options, labels, candidate_ids, selected
    )
    eligibility = pd.DataFrame(
        dict(
            hrf_id=candidate_ids,
            eligible=True,
            reason="",
            criterion="structural_random_design_validity",
        )
    )
    values = dict(
        hrf_indices=ids,
        cv_r2=None,
        canonical_cv_r2=None,
        delta_cv_r2=None,
        method="reml",
        reml=diagnostics,
        library=library,
        _eligibility=eligibility,
        run_labels=labels,
        feature_signature=signature,
        task_model=task_model,
    )
    activity = _reml_activity(
        groups, library, labels, signature, task_model, options, ids, selected
    )
    return values, activity
