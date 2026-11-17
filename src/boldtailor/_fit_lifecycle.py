"""Independent logging scopes and provenance construction for fit operations."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
from uuid import uuid4

from boldtailor.logging import (
    _make_event,
    append_event_history,
    bind_context,
    emit_event,
)
from boldtailor.provenance import software_environment
from boldtailor.provenance import ProvenanceRecord, extend_provenance


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
        completion = _make_event(
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
