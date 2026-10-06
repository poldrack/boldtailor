"""Boldtailor's b1-b4 single-trial ladder for one cortical session."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pandas as pd

from boldtailor.denoising import select_denoising
from boldtailor.fractional_ridge import (
    fraction_grid,
    score_fraction_candidates,
    select_ridge_fractions,
)
from boldtailor.hrf_library import default_hrf_library, glmsingle_hrf_library
from boldtailor.hrf_selection import select_hrfs
from boldtailor.parallel import map_blocks
from boldtailor.single_trial import fit_selected_hrfs, fit_single_trials
from boldtailor.workflow.beta_series import trial_predictors

from experiments.nsd_replication.inputs import analysis_data
from experiments.nsd_replication.lss import lss_canonical, lss_selected

LEVELS = ("b1", "b2", "b2-lib20", "b3", "b4")
LSS_LEVELS = ("lss-assume", "lss-fit")
FRACTIONS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)


@dataclass(frozen=True)
class LadderFit:
    betas: np.ndarray
    extras: dict = field(default_factory=dict)
    record: dict = field(default_factory=dict)


def _blocks(n_features, size):
    return [np.arange(s, min(s + size, n_features)) for s in range(0, n_features, size)]


def _n_features(session):
    return session.signals[0].shape[1]


def _labels(session):
    return list(session.labels)


def _stack(result):
    return np.vstack(result.run_betas)


def _onsets(result):
    return result.trial_table["onset"].to_numpy(dtype=float)


def _arrays(result, **extras):
    return dict(betas=_stack(result), onsets=_onsets(result), **extras)


def _scored_fractions():
    """The grid as fit_cv_beta_series hands it to the scorer: sorted, reversed."""
    return tuple(reversed(tuple(sorted(fraction_grid(FRACTIONS)))))


def select_session_denoising(session, task_model, library, gate=True):
    data = analysis_data(session, np.arange(_n_features(session)))
    return select_denoising(
        data,
        task_model=task_model,
        library=library,
        significance_gate=gate,
        run_labels=_labels(session),
    )


def denoised_confounds(session, result):
    names = list(result.component_names)
    return tuple(
        pd.concat([frame, pd.DataFrame(c, columns=names, index=frame.index)], axis=1)
        for frame, c in zip(session.confounds, result.run_components, strict=True)
    )


def _predictors(session, task_model):
    runs = [
        SimpleNamespace(events=e, label=label)
        for e, label in zip(session.events, session.labels)
    ]
    return trial_predictors(runs, task_model)


def _b1_block(indices, session):
    data = analysis_data(session, indices)
    return _arrays(fit_single_trials(data, run_labels=_labels(session)))


def _selection_block(indices, session, library, task_model):
    return select_hrfs(
        analysis_data(session, indices),
        library=library,
        run_labels=_labels(session),
        task_model=task_model,
    )


def _selected_block(indices, session, selection, confounds, fractions):
    result = fit_selected_hrfs(
        analysis_data(session, indices, confounds),
        hrf_selection=selection[tuple(indices)],
        run_labels=_labels(session),
        ridge_fraction=fractions,
    )
    return _arrays(result, hrf_indices=selection[tuple(indices)].hrf_indices)


def _block_signals(session, indices):
    return [np.asarray(y[:, indices], dtype=float) for y in session.signals]


def _lss_assume_block(indices, session):
    result = fit_single_trials(
        analysis_data(session, indices), run_labels=_labels(session)
    )
    betas = lss_canonical(result, _block_signals(session, indices))
    return dict(betas=betas, onsets=_onsets(result))


def _lss_fit_block(indices, session, selection):
    result = fit_selected_hrfs(
        analysis_data(session, indices),
        hrf_selection=selection[tuple(indices)],
        run_labels=_labels(session),
        ridge_fraction=None,
    )
    betas = lss_selected(result, _block_signals(session, indices))
    return dict(
        betas=betas, onsets=_onsets(result), hrf_indices=result.design.hrf_indices
    )


def _b4_block(indices, session, selection, confounds, task_model, library):
    data = analysis_data(session, indices, confounds)
    scores = score_fraction_candidates(
        data,
        _predictors(session, task_model),
        fractions=_scored_fractions(),
        library=library,
        run_labels=_labels(session),
        encoding_mode="within_run",
    )
    chosen = select_ridge_fractions(scores).ridge_fraction
    result = fit_selected_hrfs(
        data,
        hrf_selection=selection[tuple(indices)],
        run_labels=_labels(session),
        ridge_fraction=chosen,
    )
    return _arrays(result, ridge_fraction=chosen)


def _run_blocks(function, session, blocks, args, n_jobs):
    """Assemble block outputs into full trial x feature betas and feature extras."""
    n, out = _n_features(session), {}
    for indices, part in map_blocks(function, blocks, args=args, n_jobs=n_jobs):
        out.setdefault("onsets", part["onsets"])
        for key, values in part.items():
            if key != "onsets":
                shape = values.shape[:-1] + (n,)
                out.setdefault(key, np.zeros(shape, dtype=values.dtype))
                out[key][..., indices] = values
    return out


def _selections(session, blocks, library, task_model, n_jobs):
    pairs = map_blocks(
        _selection_block, blocks, args=(session, library, task_model), n_jobs=n_jobs
    )
    return {tuple(indices): selection for indices, selection in pairs}


def _denoising_record(denoising):
    return dict(
        denoise_n_components=denoising.n_components,
        denoise_pcstop_count=denoising.pcstop_count,
        denoise_gate_decision=denoising.significance_gate.decision,
    )


def _denoising_extras(denoising):
    rejected = denoising.significance_gate.decision == "rejected"
    return dict(
        denoise_n_components=np.asarray(denoising.n_components),
        denoise_pcstop_count=np.asarray(denoising.pcstop_count),
        denoise_gate_rejected=np.asarray(int(rejected)),
    )


class _Ladder:
    """Shared, lazily computed ingredients of one session's ladder."""

    def __init__(
        self, session, task_model, block_size, n_jobs, denoising, ridge_task_model
    ):
        self.session = session
        self.task_model = session.task_model if task_model is None else task_model
        self.ridge_task_model = (
            self.task_model if ridge_task_model is None else ridge_task_model
        )
        self.block_size = block_size
        self.n_jobs = n_jobs
        self.blocks = _blocks(_n_features(session), block_size)
        self.library = default_hrf_library()
        self._denoising = denoising
        self._cache = {}

    def cached(self, key, compute):
        if key not in self._cache:
            self._cache[key] = compute()
        return self._cache[key]

    def run(self, function, *args):
        return _run_blocks(function, self.session, self.blocks, args, self.n_jobs)

    def selection(self, library):
        return self.cached(
            ("selection", library.fingerprint),
            lambda: _selections(
                self.session, self.blocks, library, self.task_model, self.n_jobs
            ),
        )

    def denoising(self):
        if self._denoising is None:
            self._denoising = select_session_denoising(
                self.session, self.session.task_model, self.library
            )
        return self._denoising

    def confounds(self):
        return self.cached(
            "confounds", lambda: denoised_confounds(self.session, self.denoising())
        )

    def fit(self, level, out, library, denoised=False):
        extras = {k: v for k, v in out.items() if k not in ("betas", "onsets")}
        record = dict(
            level=level,
            n_features=_n_features(self.session),
            block_size=self.block_size,
            library_fingerprint=None if library is None else library.fingerprint,
            onsets=out["onsets"],
        )
        if denoised:
            extras.update(_denoising_extras(self.denoising()))
            record.update(_denoising_record(self.denoising()))
        return LadderFit(betas=out["betas"], extras=extras, record=record)


def _fit_b1(ladder):
    return ladder.fit("b1", ladder.run(_b1_block, ladder.session), None)


def _fit_selected(ladder, level, library):
    selection = ladder.selection(library)
    out = ladder.run(_selected_block, ladder.session, selection, None, None)
    return ladder.fit(level, out, library)


def _fit_b2(ladder):
    return _fit_selected(ladder, "b2", ladder.library)


def _fit_b2_lib20(ladder):
    return _fit_selected(ladder, "b2-lib20", glmsingle_hrf_library())


def _fit_b3(ladder):
    selection = ladder.selection(ladder.library)
    out = ladder.run(
        _selected_block, ladder.session, selection, ladder.confounds(), None
    )
    out.pop("hrf_indices")
    return ladder.fit("b3", out, ladder.library, denoised=True)


def _fit_b4(ladder):
    out = ladder.run(
        _b4_block,
        ladder.session,
        ladder.selection(ladder.library),
        ladder.confounds(),
        ladder.ridge_task_model,
        ladder.library,
    )
    return ladder.fit("b4", out, ladder.library, denoised=True)


def _fit_lss_assume(ladder):
    out = ladder.run(_lss_assume_block, ladder.session)
    return ladder.fit("lss-assume", out, None)


def _fit_lss_fit(ladder):
    selection = ladder.selection(ladder.library)
    out = ladder.run(_lss_fit_block, ladder.session, selection)
    return ladder.fit("lss-fit", out, ladder.library)


_FITTERS = {
    "lss-assume": _fit_lss_assume,
    "lss-fit": _fit_lss_fit,
    "b1": _fit_b1,
    "b2": _fit_b2,
    "b2-lib20": _fit_b2_lib20,
    "b3": _fit_b3,
    "b4": _fit_b4,
}


def _check_sorted_onsets(session):
    for label, events in zip(session.labels, session.events, strict=True):
        if not np.all(np.diff(events["onset"].to_numpy(dtype=float)) >= 0):
            raise ValueError(f"{label}: event onsets must be non-decreasing")


def fit_ladder(
    session,
    *,
    levels=LEVELS,
    task_model=None,
    block_size=4096,
    n_jobs=1,
    denoising=None,
    ridge_task_model=None,
):
    """Fit the requested levels in ladder order; betas are trials x features.

    Rows follow run order, then onset within run (as ``trial_table``); runs
    whose events are not onset-sorted raise. ``denoising`` reuses a
    ``select_session_denoising`` result for b3/b4; when selected here it always
    uses ``session.task_model``, whatever ``task_model`` is passed.
    ``ridge_task_model`` supplies b4's encoding predictors (default: the task
    model in effect); encoding ridge CV needs at least one modulator.
    """
    unknown = set(levels) - set(_FITTERS)
    if unknown:
        raise ValueError(f"unknown ladder levels: {sorted(unknown)}")
    _check_sorted_onsets(session)
    ladder = _Ladder(
        session, task_model, block_size, n_jobs, denoising, ridge_task_model
    )
    return {
        level: _FITTERS[level](ladder)
        for level in (*LEVELS, *LSS_LEVELS)
        if level in levels
    }
