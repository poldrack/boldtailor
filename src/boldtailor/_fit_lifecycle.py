"""Execution bookkeeping around explicit scientific fit operations."""

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
from uuid import uuid4

from boldtailor.logging import (
    _emit_record,
    _make_event,
    append_event_history,
    bind_context,
    emit_event,
)
from boldtailor._software import software_environment
from boldtailor.provenance import ProvenanceRecord, extend_provenance


@dataclass
class FitOperation:
    name: str
    parent: ProvenanceRecord
    execution_id: str = field(default_factory=lambda: str(uuid4()))
    analysis_id: str | None = None
    history: tuple[Mapping[str, object], ...] = ()
    completed: Mapping[str, object] | None = None

    def provenance(
        self,
        activity: Mapping[str, object],
        *,
        analysis_id: str | None,
        warnings: Sequence[Mapping[str, object]] = (),
    ) -> ProvenanceRecord:
        """Prepare final provenance; completion is logged on successful exit."""
        if self.completed is not None:
            raise RuntimeError("fit operation already has completed provenance")
        self.analysis_id = analysis_id
        completion = _make_event(
            f"{self.name}_completed", stage="fit", analysis_id=analysis_id
        )
        record = extend_provenance(
            self.parent,
            execution_id=self.execution_id,
            activity={**activity, "software": software_environment()},
            events=append_event_history(self.history, completion),
            warnings=warnings,
            analysis_id=analysis_id,
        )
        self.completed = completion
        return record


@contextmanager
def fit_operation(name: str, parent: ProvenanceRecord) -> Iterator[FitOperation]:
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
            if operation.completed is None:
                raise RuntimeError("fit operation has no completed provenance")
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
            _emit_record(operation.completed)
