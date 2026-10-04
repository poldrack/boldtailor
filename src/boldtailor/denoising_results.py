"""Owned results of task-guided denoising selection.

Scores and fold diagnostics are selection statistics used to choose one PC
count; they are not independent performance estimates. The final noise pool
and components come from all supplied runs and are meant for final fitting.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from boldtailor._arrays import own_array_tuples, own_fields, own_tuples, rebind
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
    """Training-only HRFs, masks, and PCA of one count-selection fold.

    ``pool_threshold`` is the threshold applied to ``pool_r2``; with
    ``pool_r2_threshold="auto"`` it was fitted on this fold's training
    statistic and ``pool_mixture`` holds the fit (else ``None``).
    """

    validation_run: str
    training_runs: tuple[str, ...]
    pool: np.ndarray
    scoring: np.ndarray
    scored: np.ndarray
    zero_target: np.ndarray
    pool_r2: np.ndarray
    hrf_indices: np.ndarray
    pool_threshold: float
    pool_mixture: MixtureThreshold | None
    components: PcaDiagnostics

    def __post_init__(self):
        own_fields(self, ("pool", "scoring", "scored", "zero_target"), dtype=bool)
        own_fields(self, ("pool_r2",))
        own_fields(self, ("hrf_indices",), dtype=np.int64)
        own_tuples(self, ("training_runs",))


@dataclass(frozen=True, kw_only=True, eq=False)
class DenoisingResult:
    """Chosen PC count, final full-data pool and run PCs, and diagnostics.

    ``pool_r2`` is the final initial selection's indicator-consistent
    leave-one-run-out task-model R² that defined ``noise_pool``;
    ``selection_cv_r2`` is the raw ``select_hrfs`` score.
    ``pool_r2_threshold`` is the setting (``"auto"`` or a float);
    ``noise_pool_threshold`` is the value applied to the final ``pool_r2``,
    and ``noise_pool_mixture`` the final mixture fit when automatic.
    """

    n_components: int
    counts: tuple[int, ...]
    pool_r2_threshold: float | str
    score_tolerance: float
    noise_pool_threshold: float
    noise_pool_mixture: MixtureThreshold | None
    noise_pool: np.ndarray
    scoring_mask: np.ndarray
    pool_r2: np.ndarray
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
        own_fields(self, ("noise_pool", "scoring_mask"), dtype=bool)
        own_fields(self, ("pool_r2",))
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
        for label, comps, run in zip(
            self.run_labels, self.run_components, self.source_identity
        ):
            if comps.shape[0] != run.n_scans:
                raise ValueError(
                    f"run_components for run '{label}' has {comps.shape[0]} rows; "
                    f"the run has {run.n_scans} scans"
                )
        if self.n_components not in self.counts or any(
            c.ndim != 2 or c.shape[1] != self.n_components for c in self.run_components
        ):
            raise ValueError(
                "n_components must be a candidate count and match every "
                "run_components array"
            )

    @property
    def candidate_scores(self) -> pd.DataFrame:
        """Per count: eligibility, equal-weight fold-mean R², and reasons."""
        return _owned_table(self._candidate_scores)

    @property
    def fold_scores(self) -> pd.DataFrame:
        """Per fold and count: eligibility, reason, mean R², and feature counts."""
        return _owned_table(self._fold_scores)

    @property
    def component_names(self) -> tuple[str, ...]:
        return component_names(self.n_components)

    @property
    def selection_cv_r2(self) -> np.ndarray:
        return self.initial_selection.cv_r2

    @property
    def initial_hrf_indices(self) -> np.ndarray:
        return self.initial_selection.hrf_indices
