"""Metric tables on disk: written one at a time, skipped when present."""

import pandas as pd


def _path(out, name):
    return out / f"{name}.tsv"


def ensure_tables(out, builders, data, recompute=False):
    """Build and write each table in turn; existing tables are read back
    instead of rebuilt unless ``recompute``."""
    out.mkdir(parents=True, exist_ok=True)
    tables = {}
    for name, build in builders.items():
        path = _path(out, name)
        if path.is_file() and not recompute:
            tables[name] = pd.read_csv(path, sep="\t")
            continue
        tables[name] = build(data)
        tables[name].to_csv(path, sep="\t", index=False)
    return tables
