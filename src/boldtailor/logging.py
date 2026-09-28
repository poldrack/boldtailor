from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import itertools
import json
import logging
from types import MappingProxyType

import numpy as np

_LOGGER_NAME = "boldtailor"
_EVENT_HISTORY_LIMIT = 8
_SEQUENCE = itertools.count(1)
_LOG_CONTEXT: ContextVar[Mapping[str, object]] = ContextVar(
    "boldtailor_log_context",
    default=MappingProxyType({}),
)


@contextmanager
def bind_context(
    *,
    execution_id: str | None = None,
    data_id: str | None = None,
    analysis_id: str | None = None,
    run_index: int | None = None,
    inherit: bool = True,
) -> Iterator[None]:
    updates = _context_updates(
        execution_id=execution_id,
        data_id=data_id,
        analysis_id=analysis_id,
        run_index=run_index,
    )
    parent = _LOG_CONTEXT.get() if inherit else {}
    token = _LOG_CONTEXT.set({**parent, **updates})
    try:
        yield
    finally:
        _LOG_CONTEXT.reset(token)


def emit_event(
    event: str,
    *,
    stage: str,
    level: int = logging.INFO,
    error: Exception | str | None = None,
    execution_id: str | None = None,
    data_id: str | None = None,
    analysis_id: str | None = None,
    run_index: int | None = None,
) -> Mapping[str, object]:
    payload = _make_event(
        event=event,
        stage=stage,
        level=level,
        error=error,
        execution_id=execution_id,
        data_id=data_id,
        analysis_id=analysis_id,
        run_index=run_index,
    )
    _emit_record(payload, level=level)
    return payload


def _emit_record(payload: Mapping[str, object], *, level: int = logging.INFO) -> None:
    logging.getLogger(_LOGGER_NAME).log(
        level,
        json.dumps(
            dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ),
    )


def append_event_history(
    history: Sequence[Mapping[str, object]],
    event: Mapping[str, object],
) -> tuple[Mapping[str, object], ...]:
    updated = tuple(history) + (_history_entry(event),)
    if len(updated) <= _EVENT_HISTORY_LIMIT:
        return updated
    return updated[-_EVENT_HISTORY_LIMIT:]


def _context_updates(
    *,
    execution_id: str | None,
    data_id: str | None,
    analysis_id: str | None,
    run_index: int | None,
) -> dict[str, object]:
    updates: dict[str, object] = {}
    if execution_id is not None:
        updates["execution_id"] = execution_id
    if data_id is not None:
        updates["data_id"] = data_id
    if analysis_id is not None:
        updates["analysis_id"] = analysis_id
    if run_index is not None:
        updates["run_index"] = run_index
    return updates


def _make_event(
    event: str,
    *,
    stage: str,
    level: int = logging.INFO,
    error: Exception | str | None = None,
    execution_id: str | None = None,
    data_id: str | None = None,
    analysis_id: str | None = None,
    run_index: int | None = None,
) -> Mapping[str, object]:
    payload: dict[str, object] = {
        "timestamp": _timestamp(),
        "sequence": next(_SEQUENCE),
        "level": logging.getLevelName(level),
        "event": event,
        "stage": stage,
    }
    payload.update(_LOG_CONTEXT.get())
    payload.update(
        _context_updates(
            execution_id=execution_id,
            data_id=data_id,
            analysis_id=analysis_id,
            run_index=run_index,
        )
    )
    if error is not None:
        payload["error_code"] = _error_code(error)
    return MappingProxyType(payload)


def _history_entry(event: Mapping[str, object]) -> Mapping[str, object]:
    keep = (
        "timestamp",
        "sequence",
        "level",
        "event",
        "stage",
        "execution_id",
        "data_id",
        "analysis_id",
        "run_index",
        "error_code",
    )
    return {key: event[key] for key in keep if key in event}


def _timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace(
            "+00:00",
            "Z",
        )
    )


def _error_code(error: Exception | str) -> str:
    if isinstance(error, (ArithmeticError, np.linalg.LinAlgError)):
        return "numerical_failure"
    if isinstance(error, (ValueError, TypeError)):
        return "invalid_input"
    if isinstance(error, OSError):
        return "io_failure"
    return "operation_failed"
