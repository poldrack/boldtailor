"""Detect a task model from event tables: response time and categorical trial type."""

from boldtailor.model import Modulator, TaskModel, level_name

TRIAL_TYPE_NOTE = "trial_type has fewer than two levels; not used as a modulator"


def _run_labels(tables, labels):
    return list(labels) if labels else [f"run {i}" for i in range(1, len(tables) + 1)]


def _table_levels(table, column, label):
    try:
        return {level_name(v) for v in table[column]}
    except ValueError as error:
        raise ValueError(f"{label}: {column!r}: {error}") from error


def observed_levels(tables, column, labels=None):
    """Canonical non-missing level names of ``column`` across every table."""
    names = set()
    for table, label in zip(tables, _run_labels(tables, labels), strict=True):
        names |= _table_levels(table, column, label)
    names.discard(None)
    return names


def _everywhere(column, tables):
    return bool(tables) and all(column in t.columns for t in tables)


def _automatic(tables, labels):
    modulators = []
    if _everywhere("response_time", tables):
        modulators.append(Modulator("response_time", missing="indicator"))
    if _everywhere("trial_type", tables):
        levels = observed_levels(tables, "trial_type", labels)
        if len(levels) >= 2:
            modulators.append(
                Modulator("trial_type", kind="categorical", levels=tuple(levels))
            )
    return tuple(modulators)


def _resolved(modulator, tables, labels):
    if modulator.resolved:
        return modulator
    try:
        return modulator.with_levels(observed_levels(tables, modulator.column, labels))
    except ValueError as error:
        raise ValueError(f"modulator {modulator.column!r}: {error}") from error


def _check_columns(tables, labels, modulators):
    for label, table in zip(labels, tables, strict=True):
        missing = [m.column for m in modulators if m.column not in table.columns]
        if missing:
            raise ValueError(f"{label} events lack modulator column(s) {missing}")


def detect_task_model(events_tables, modulators=None, labels=None):
    """The task model from event columns, or explicit modulators checked per run.

    Detection takes ``response_time`` and a categorical ``trial_type`` with at
    least two levels; unresolved categorical modulators take their levels from
    every run. ``labels`` name the runs in errors.
    """
    tables = list(events_tables)
    labels = _run_labels(tables, labels)
    if modulators is None:
        modulators = _automatic(tables, labels)
    _check_columns(tables, labels, modulators)
    return TaskModel(tuple(_resolved(m, tables, labels) for m in modulators))


def task_model_notes(events_tables, modulators=None):
    """Why automatic detection left out a column that every run has."""
    tables = list(events_tables)
    if modulators is None and _everywhere("trial_type", tables):
        if len(observed_levels(tables, "trial_type")) < 2:
            return [TRIAL_TYPE_NOTE]
    return []
