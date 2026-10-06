"""Metric tables on disk: written one at a time, skipped when present.

Tables record what they were computed from: the levels that formed the
composite reliability (``composite_levels``) and the session window
(``sessions``), each "|"-joined. An existing table computed from a different
level set or session window is never silently reused or replaced.
"""

import pandas as pd

COMPOSITE_COLUMN = "composite_levels"
SESSIONS_COLUMN = "sessions"


def _path(out, name):
    return out / f"{name}.tsv"


def _stored(path, column):
    table = pd.read_csv(path, sep="\t")
    if column not in table or table[column].nunique() != 1:
        return None
    return set(str(table[column].iloc[0]).split("|"))


def stored_levels(path):
    """The composite level set recorded in a table, or None if unrecorded."""
    return _stored(path, COMPOSITE_COLUMN)


def _check(path, column, wanted):
    stored = _stored(path, column)
    if stored != set(wanted):
        found = "unrecorded" if stored is None else "|".join(sorted(stored))
        raise ValueError(
            f"{path} was computed with {column} {found}, not "
            f"{'|'.join(wanted)}; pass --recompute to overwrite it"
        )


def _recorded(levels, sessions):
    return {COMPOSITE_COLUMN: levels, SESSIONS_COLUMN: sessions}


def _check_existing(out, names, recorded, recompute):
    if recompute:
        return
    for name in names:
        if not _path(out, name).is_file():
            continue
        for column, wanted in recorded.items():
            if wanted is not None:
                _check(_path(out, name), column, wanted)


def _build(build, data, recorded):
    table = build(data)
    columns = {c: "|".join(v) for c, v in recorded.items() if v is not None}
    return table.assign(**columns)


def ensure_tables(out, builders, data, recompute=False, levels=None, sessions=None):
    """Build and write each table in turn; existing tables are read back
    instead of rebuilt unless ``recompute``. With ``levels`` (the composite's
    level set) and ``sessions`` (the session window) every table records them,
    and existing tables computed from different ones raise before anything is
    built."""
    out.mkdir(parents=True, exist_ok=True)
    recorded = _recorded(levels, sessions)
    _check_existing(out, builders, recorded, recompute)
    tables = {}
    for name, build in builders.items():
        path = _path(out, name)
        if path.is_file() and not recompute:
            tables[name] = pd.read_csv(path, sep="\t")
            continue
        tables[name] = _build(build, data, recorded)
        tables[name].to_csv(path, sep="\t", index=False)
    return tables
