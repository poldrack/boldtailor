"""Identify global tuning decisions and connect final beta fits to them."""

from hashlib import sha256
from uuid import uuid4

import numpy as np

from boldtailor.provenance import (
    ProvenanceRecord,
    analysis_fingerprint,
    extend_provenance,
)


def tuning_provenance(scores, selection):
    activity = dict(
        name="global_encoding_ridge_selection",
        run_labels=list(scores.run_labels),
        alphas=list(selection.alphas),
        percentile=selection.percentile,
        objective="percentile_of_pooled_within_run_trial_encoding_r2",
        validation_target="candidate_regularized_betas",
        tie_rule="smallest alpha within 1e-12 of the maximum objective",
        scoring_mask_fingerprint=sha256(selection.scoring_mask.tobytes()).hexdigest(),
        candidate_score_fingerprint=sha256(
            np.asarray(scores.cv_r2, dtype="<f8").tobytes()
        ).hexdigest(),
        objective_scores=selection.objective_scores.tolist(),
        selected_alpha=selection.ridge_alpha,
    )
    record = scores.provenance
    identity = dict(scoring=record.to_dict()["activities"], selection=activity)
    return extend_provenance(
        record,
        execution_id=str(uuid4()),
        activity=activity,
        events=record.events,
        warnings=(),
        analysis_id=analysis_fingerprint(record.metadata_fingerprint, identity),
    )


def link_final_provenance(result, decision):
    """Keep block fitting records, adding a reference to the saved decision."""
    for block in result["provenance"]:
        record = ProvenanceRecord.from_dict(block["record"])
        identity = dict(
            fitting_analysis_fingerprint=record.analysis_fingerprint,
            tuning_analysis_fingerprint=decision.analysis_fingerprint,
        )
        activity = dict(
            name="encoding_guided_ridge_refit",
            tuning_scope="all",
            selected_alpha=result["ridge_alpha"],
            tuning_execution_id=decision.execution_id,
            **identity,
        )
        block["record"] = extend_provenance(
            record,
            execution_id=str(uuid4()),
            activity=activity,
            events=record.events,
            warnings=(),
            analysis_id=analysis_fingerprint(record.metadata_fingerprint, identity),
        ).to_dict()
