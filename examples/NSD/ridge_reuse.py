"""Restore ridge tuning and outer-score summaries from published arrays."""

import numpy as np

from boldtailor.fractional_ridge import select_ridge_fractions
from boldtailor.ridge_selection import select_ridge_penalty
from boldtailor.ridge_results import CandidateScores
from boldtailor.provenance import ProvenanceRecord
from boldtailor.single_trial import r_squared
from .workflow_reuse import read_json, read_map


def _tuning(base, descriptor, brain, fractional):
    prefix = f"{base}_desc-notebook{descriptor}"
    meta = read_json(prefix + "_metadata.json")
    grid = meta["fractions" if fractional else "alphas"]
    with np.load(prefix + "_folds.npz") as archive:
        sse, sst = archive["sse"], archive["sst"]
    scores = r_squared(sse.sum(axis=0), sst.sum(axis=0))
    ids = read_map(base, descriptor, "foldhrfindex", brain)
    provenance = ProvenanceRecord.from_dict(read_json(prefix + "_provenance.json"))
    candidates = CandidateScores(
        regularization="fractional_ridge" if fractional else "normalized_ridge",
        grid=grid,
        cv_r2=scores,
        fold_sse=sse,
        fold_sst=sst,
        fold_hrf_indices=np.where(np.isfinite(ids), ids, -1).astype(int),
        trial_masks=meta["trial_masks"],
        run_labels=meta["run_labels"],
        provenance=provenance,
    )
    percentile = meta["summary_percentile" if fractional else "percentile"]
    selected = (
        select_ridge_fractions(scores, grid)
        if fractional
        else select_ridge_penalty(scores, grid, percentile=percentile)
    )
    return dict(scores=candidates, selection=selected, summary_percentile=percentile)


def _outer(base, descriptor, brain, fractional):
    meta = read_json(f"{base}_desc-notebook{descriptor}_metadata.json")
    result = dict(
        encoding_r2=read_map(base, descriptor, "encodingpredictionr2", brain)[0],
        ridge_alpha=meta["ridge_alpha"],
    )
    if fractional:
        result["ridge_fraction"] = read_map(base, descriptor, "ridgefraction", brain)[0]
    return result


def load_ridge_summaries(base, brain, settings):
    fractional = settings["ridge_mode"] == "fractional_cv"
    kind = "FractionalCV" if fractional else "RidgeCV"
    return {
        mode: dict(
            tuning={
                scope: _tuning(base, mode + kind + scope.title(), brain, fractional)
                for scope in ("odd", "even", "all")
            },
            evaluation={
                scope: _outer(
                    base,
                    mode + kind + "".join(w.title() for w in scope.split("_")),
                    brain,
                    fractional,
                )
                for scope in ("odd_to_even", "even_to_odd")
            },
        )
        for mode in ("Canonical", "Optimized")
    }
