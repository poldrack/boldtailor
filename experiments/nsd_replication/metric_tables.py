"""Metric tables on disk: written one at a time, skipped when present.

Tables built on a composite reliability record the levels that formed it in
a ``composite_levels`` column ("|"-joined); an existing table built from a
different level set is never silently replaced.
"""

import pandas as pd

COMPOSITE_COLUMN = "composite_levels"


def _path(out, name):
    return out / f"{name}.tsv"


def stored_levels(path):
    """The composite level set recorded in a table, or None if unrecorded."""
    table = pd.read_csv(path, sep="\t")
    if COMPOSITE_COLUMN not in table or table[COMPOSITE_COLUMN].nunique() != 1:
        return None
    return set(str(table[COMPOSITE_COLUMN].iloc[0]).split("|"))


def _check_levels(path, levels):
    stored = stored_levels(path)
    if stored != set(levels):
        found = "unrecorded" if stored is None else "|".join(sorted(stored))
        raise ValueError(
            f"{path} was computed with composite levels {found}, not "
            f"{'|'.join(levels)}; pass --recompute to overwrite it"
        )


def _check_existing(out, names, levels, recompute):
    if recompute or levels is None:
        return
    for name in names:
        if _path(out, name).is_file():
            _check_levels(_path(out, name), levels)


def _build(build, data, levels):
    table = build(data)
    if levels is None:
        return table
    return table.assign(**{COMPOSITE_COLUMN: "|".join(levels)})


def ensure_tables(out, builders, data, recompute=False, levels=None):
    """Build and write each table in turn; existing tables are read back
    instead of rebuilt unless ``recompute``. With ``levels`` (the composite's
    level set) every table records it, and existing tables computed from a
    different set raise before anything is built."""
    out.mkdir(parents=True, exist_ok=True)
    _check_existing(out, builders, levels, recompute)
    tables = {}
    for name, build in builders.items():
        path = _path(out, name)
        if path.is_file() and not recompute:
            tables[name] = pd.read_csv(path, sep="\t")
            continue
        tables[name] = _build(build, data, levels)
        tables[name].to_csv(path, sep="\t", index=False)
    return tables
