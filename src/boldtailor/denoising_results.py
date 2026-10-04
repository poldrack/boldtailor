"""Owned results of task-guided denoising selection.

Scores and fold diagnostics are selection statistics used to choose one PC
count; they are not independent performance estimates. HRFs, the noise pool,
and its components come from all supplied runs (as in GLMsingle) and are
meant for final fitting.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from boldtailor._arrays import (
    own_array_tuples,
    own_fields,
    own_tuples,
    readonly_array,
    rebind,
)
from boldtailor._denoising_identity import RunIdentity
from boldtailor._mixture_threshold import MixtureThreshold
from boldtailor.data import _owned_table
from boldtailor.hrf_results import HrfSelectionResult
from boldtailor.provenance import ProvenanceRecord

COMPONENT_PREFIX = "denoise_pc_"


def component_names(count: int) -> tuple[str, ...]:
    return tuple(f"{COMPONENT_PREFIX}{i:03d}" for i in range(count))


@dataclass(frozen=True, kw_only=True, eq=False)
class PcaDiagnostics:
    """Per-run normalized pool PCA: effective ranks and all singular values."""

    run_labels: tuple[str, ...]
    pool_size: int
    ranks: tuple[int, ...]
    singular_values: tuple[np.ndarray, ...]
    rank_tolerances: tuple[float, ...]
    retained_columns: tuple[int, ...]

    def __post_init__(self):
        own_tuples(self, ("run_labels", "ranks", "rank_tolerances", "retained_columns"))
        own_array_tuples(self, ("singular_values",))

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(
            dict(
                run_label=self.run_labels,
                pool_size=self.pool_size,
                retained_columns=self.retained_columns,
                rank=self.ranks,
                rank_tolerance=self.rank_tolerances,
            )
        )


@dataclass(frozen=True, kw_only=True, eq=False)
class DenoisingFold:
    """One count-selection fold: its runs and the held-out targets.

    HRFs, the pool, its PCs, and the scoring features are full-data and
    shared by every fold. ``zero_target`` marks numerically zero held-out
    targets; ``target_energy`` is the held-out target energy (NaN for
    unscored features).
    """

    validation_run: str
    training_runs: tuple[str, ...]
    zero_target: np.ndarray
    target_energy: np.ndarray

    def __post_init__(self):
        own_fields(self, ("zero_target",), dtype=bool)
        own_fields(self, ("target_energy",))
        own_tuples(self, ("training_runs",))


@dataclass(frozen=True, kw_only=True, eq=False)
class DenoisingResult:
    """Chosen PC count, full-data pool and run PCs, and diagnostics.

    ``onoff_r2`` is GLMsingle's ON-OFF R² that defined ``noise_pool`` (below
    ``noise_pool_threshold``) and ``scoring_mask`` (above it, or the 100
    best features when ``scoring_fallback``). ``scored`` is the scoring mask
    minus features with a zero held-out target. ``pool_r2_threshold`` is the
    setting (``"auto"`` or a float) and ``noise_pool_mixture`` the mixture
    fit when automatic. ``initial_selection`` holds the frozen full-data
    HRFs. ``perf``/``curve`` are the per-count median performance and its
    gain over zero PCs (NaN when unavailable).
    """

    n_components: int
    counts: tuple[int, ...]
    pool_r2_threshold: float | str
    pcstop: float
    noise_pool_threshold: float
    noise_pool_mixture: MixtureThreshold | None
    noise_pool: np.ndarray
    scoring_mask: np.ndarray
    scoring_fallback: bool
    scored: np.ndarray
    onoff_r2: np.ndarray
    initial_selection: HrfSelectionResult
    run_components: tuple[np.ndarray, ...]
    components: PcaDiagnostics
    folds: tuple[DenoisingFold, ...]
    _candidate_scores: pd.DataFrame
    _fold_scores: pd.DataFrame
    run_labels: tuple[str, ...]
    feature_signature: str | None
    source_identity: tuple[RunIdentity, ...]
    provenance: ProvenanceRecord

    def __post_init__(self):
        own_fields(self, ("noise_pool", "scoring_mask", "scored"), dtype=bool)
        own_fields(self, ("onoff_r2",))
        own_array_tuples(self, ("run_components",))
        own_tuples(self, ("counts", "folds", "run_labels", "source_identity"))
        rebind(
            self,
            _candidate_scores=_owned_table(self._candidate_scores),
            _fold_scores=_owned_table(self._fold_scores),
        )
        self._validate()

    def _validate(self):
        if not isinstance(self.initial_selection, HrfSelectionResult):
            raise ValueError("initial_selection must be an HrfSelectionResult")
        if (
            not len(self.run_components)
            == len(self.run_labels)
            == len(self.source_identity)
        ):
            raise ValueError("run_components must hold one array per run")
        if self.n_components not in self.counts or any(
            c.ndim != 2 or c.shape[1] != self.n_components for c in self.run_components
        ):
            raise ValueError(
                "n_components must be a candidate count and match every "
                "run_components array"
            )
        for label, comps, run in zip(
            self.run_labels, self.run_components, self.source_identity
        ):
            if comps.shape[0] != run.n_scans:
                raise ValueError(
                    f"run_components for run '{label}' has {comps.shape[0]} rows; "
                    f"the run has {run.n_scans} scans"
                )

    @property
    def candidate_scores(self) -> pd.DataFrame:
        """Per count: eligibility, median performance, curve, and reasons."""
        return _owned_table(self._candidate_scores)

    @property
    def fold_scores(self) -> pd.DataFrame:
        """Per fold and count: eligibility, reason, fold median R², sizes."""
        return _owned_table(self._fold_scores)

    @property
    def perf(self) -> np.ndarray:
        return readonly_array(self._candidate_scores["perf"].to_numpy(dtype=float))

    @property
    def curve(self) -> np.ndarray:
        return readonly_array(self._candidate_scores["curve"].to_numpy(dtype=float))

    @property
    def component_names(self) -> tuple[str, ...]:
        return component_names(self.n_components)

    @property
    def selection_cv_r2(self) -> np.ndarray:
        return self.initial_selection.cv_r2

    @property
    def initial_hrf_indices(self) -> np.ndarray:
        return self.initial_selection.hrf_indices
