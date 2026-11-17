"""Provenance of ridge tuning decisions and of the refit that follows them."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256
from uuid import uuid4

import numpy as np

from boldtailor._regularizers import NORMALIZED_RIDGE, regularizer_named
from boldtailor.fractional_ridge import NORM_BASIS
from boldtailor.provenance import (
    ProvenanceRecord,
    analysis_fingerprint,
    extend_provenance,
)
from boldtailor.trial_encoding import encoding_metadata
from boldtailor.workflow.records import BetaFit


def cv_provenance(
    regularizer, grid, final_penalty, *, encoding_mode, percentile, library
):
    """How one CV beta series was tuned, evaluated and refit."""
    provenance = dict(
        **encoding_metadata(encoding_mode),
        objective="percentile_of_" + encoding_metadata(encoding_mode)["score"],
        percentile=percentile,
        alphas=list(grid),
        validation_target=regularizer.validation_target,
        hrf_model="canonical" if library is None else "optimized",
        library_fingerprint=None if library is None else library.fingerprint,
        final_ridge_alpha=None if regularizer.per_feature else final_penalty,
        outer_splits="odd_to_even_and_even_to_odd",
    )
    if regularizer.per_feature:
        provenance.update(
            objective="maximum_encoding_r2_per_grayordinate",
            fractions=provenance.pop("alphas"),
            validation_target=regularizer.validation_target,
            percentile_role="descriptive_only",
            fraction_norm_basis=NORM_BASIS,
        )
    return provenance


def tuning_provenance(scores, selection, *, encoding_mode):
    """Record a tuning decision with the explicitly supplied scoring mode."""
    family = regularizer_named(scores.regularization)
    per_feature = family.per_feature
    if per_feature:
        activity: dict[str, object] = dict(
            name="voxelwise_encoding_fraction_selection",
            fractions=list(selection.fractions),
            objective="maximum_encoding_r2_per_grayordinate",
            validation_target=family.validation_target,
            fraction_norm_basis=NORM_BASIS,
            selected_fraction_fingerprint=sha256(
                np.asarray(selection.ridge_fraction, dtype="<f8").tobytes()
            ).hexdigest(),
            tie_rule=family.tie_rule,
        )
    else:
        activity = _alpha_activity(selection)
    activity.update(encoding_metadata(encoding_mode))
    if not per_feature:
        activity["objective"] = (
            "percentile_of_" + encoding_metadata(encoding_mode)["score"]
        )
    activity.update(
        run_labels=list(scores.run_labels),
        scoring_mask_fingerprint=sha256(selection.scoring_mask.tobytes()).hexdigest(),
        candidate_score_fingerprint=sha256(
            np.asarray(scores.cv_r2, dtype="<f8").tobytes()
        ).hexdigest(),
    )
    record = scores.provenance
    identity = dict(scoring=_identity_activities(record), selection=activity)
    return extend_provenance(
        record,
        execution_id=str(uuid4()),
        activity=activity,
        events=record.events,
        warnings=(),
        analysis_fingerprint=analysis_fingerprint(record.data_fingerprint, identity),
    )


def _identity_activities(record):
    """Embedded scoring activities without environment-only software keys."""
    return [
        {key: value for key, value in activity.items() if key != "software"}
        for activity in record.activity_records()
    ]


def _alpha_activity(selection) -> dict[str, object]:
    return dict(
        name="global_encoding_ridge_selection",
        alphas=list(selection.alphas),
        percentile=selection.percentile,
        objective="percentile_of_pooled_within_run_trial_encoding_r2",
        validation_target=NORMALIZED_RIDGE.validation_target,
        tie_rule=NORMALIZED_RIDGE.tie_rule,
        objective_scores=selection.objective_scores.tolist(),
        selected_alpha=selection.ridge_alpha,
    )


def link_final_provenance(result, decision, *, encoding_mode) -> BetaFit:
    """Return a fit with new block records referencing the all-run decision.

    Provenance is copied; numerical arrays are shared and are not modified.
    Pass an unlinked fit: linking an already linked result raises ValueError.
    """
    blocks = deepcopy(list(result.provenance))
    for block in blocks:
        record = ProvenanceRecord.from_dict(block["record"])
        if any(
            activity.get("name") == "encoding_guided_ridge_refit"
            for activity in record.activity_records()
        ):
            raise ValueError("final fit provenance is already linked")
        identity = dict(
            fitting_analysis_fingerprint=record.analysis_fingerprint,
            tuning_analysis_fingerprint=decision.analysis_fingerprint,
        )
        activity = dict(
            name="encoding_guided_ridge_refit",
            **encoding_metadata(encoding_mode),
            tuning_scope="all",
            selected_alpha=result.ridge_alpha,
            tuning_execution_id=decision.execution_id,
            **identity,
        )
        if result.fractional:
            activity.update(
                regularization="fractional_ridge", fraction_norm_basis=NORM_BASIS
            )
        block["record"] = extend_provenance(
            record,
            execution_id=str(uuid4()),
            activity=activity,
            events=record.events,
            warnings=(),
            analysis_fingerprint=analysis_fingerprint(
                record.data_fingerprint, identity
            ),
        ).to_dict()
    return replace(result, provenance=tuple(blocks))
