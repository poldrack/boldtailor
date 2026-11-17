"""Independent logging scopes and provenance construction for fit operations."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
from uuid import uuid4

from boldtailor.logging import (
    append_event_history,
    bind_context,
    emit_event,
    emit_record,
    make_event,
)
from boldtailor.provenance import (
    ProvenanceRecord,
    analysis_fingerprint,
    extend_provenance,
    software_environment,
)


@dataclass
class FitOperation:
    """Execution metadata; analysis_id may be set when the model identity is known."""

    name: str
    parent: ProvenanceRecord
    execution_id: str = field(default_factory=lambda: str(uuid4()))
    analysis_id: str | None = None
    history: tuple[Mapping[str, object], ...] = ()


def build_fit_provenance(
    operation: FitOperation,
    activity: Mapping[str, object],
    *,
    warnings: Sequence[Mapping[str, object]] = (),
) -> ProvenanceRecord:
    """Build a snapshot without mutating the operation or emitting log messages.

    The snapshot has its own completion timestamp and sequence number. The
    logging scope independently records completion on normal exit, including
    when no provenance is requested. This builder can be called more than once
    or after the scope has exited; neither changes the scope's outcome.
    """
    with bind_context(
        execution_id=operation.execution_id,
        data_id=operation.parent.metadata_fingerprint,
        inherit=False,
    ):
        completion = make_event(
            f"{operation.name}_completed",
            stage="fit",
            analysis_id=operation.analysis_id,
        )
    return extend_provenance(
        operation.parent,
        execution_id=operation.execution_id,
        activity={**activity, "software": software_environment()},
        events=append_event_history(operation.history, completion),
        warnings=warnings,
        analysis_id=operation.analysis_id,
    )


def finish_fit(
    operation: FitOperation,
    activity: Mapping[str, object],
    *,
    warnings: Sequence[Mapping[str, object]] = (),
) -> ProvenanceRecord:
    """Identify the operation by its parent data and activity, then record it.

    For operations whose analysis identity is the fingerprint of the activity
    itself; callers with another identity set ``analysis_id`` themselves.
    """
    operation.analysis_id = analysis_fingerprint(
        operation.parent.metadata_fingerprint, activity
    )
    return build_fit_provenance(operation, activity, warnings=warnings)


@contextmanager
def fit_operation(name: str, parent: ProvenanceRecord) -> Iterator[FitOperation]:
    """Log start, then success on normal exit or failure on an exception.

    No methods need to be called on the yielded metadata. In particular,
    provenance construction is independent of the logging scope.
    """
    operation = FitOperation(name, parent)
    with bind_context(
        execution_id=operation.execution_id,
        data_id=parent.metadata_fingerprint,
        inherit=False,
    ):
        operation.history = append_event_history(
            parent.events, emit_event(f"{name}_started", stage="fit")
        )
        try:
            yield operation
        except Exception as error:
            emit_event(
                f"{name}_failed",
                stage="fit",
                level=logging.ERROR,
                error=error,
                analysis_id=operation.analysis_id,
            )
            raise
        else:
            emit_event(
                f"{name}_completed", stage="fit", analysis_id=operation.analysis_id
            )


@dataclass
class Normalization:
    """Execution metadata of one input normalization; set ``data_id`` once known."""

    stage: str
    execution_id: str = field(default_factory=lambda: str(uuid4()))
    data_id: str | None = None
    history: tuple[Mapping[str, object], ...] = ()
    completed: Mapping[str, object] | None = None

    def provenance(self, sources, activity) -> ProvenanceRecord:
        """The record of the normalized inputs; its last event is the logged completion."""
        self.completed = make_event(
            "normalization_completed", stage=self.stage, data_id=self.data_id
        )
        return ProvenanceRecord(
            execution_id=self.execution_id,
            sources=sources,
            activities=(activity,),
            events=append_event_history(self.history, self.completed),
        )


@contextmanager
def normalization(stage: str) -> Iterator[Normalization]:
    """Log start, then the recorded completion on normal exit or a failure.

    Unlike a fit, normalization logs the very completion event stored in its
    provenance, so build the provenance inside the scope.
    """
    operation = Normalization(stage)
    with bind_context(execution_id=operation.execution_id, inherit=False):
        operation.history = append_event_history(
            (), emit_event("normalization_started", stage=stage)
        )
        try:
            yield operation
        except Exception as error:
            emit_event(
                "normalization_failed",
                stage=stage,
                level=logging.ERROR,
                error=error,
                data_id=operation.data_id,
            )
            raise
        emit_record(
            operation.completed
            or make_event(
                "normalization_completed", stage=stage, data_id=operation.data_id
            )
        )
