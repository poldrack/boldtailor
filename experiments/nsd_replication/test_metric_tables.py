import pandas as pd

from experiments.nsd_replication.metric_tables import ensure_tables


def test_ensure_tables_logs_built_and_kept_tables(tmp_path, caplog):
    caplog.set_level("INFO", logger="experiments.nsd_replication")
    builders = {"a": lambda _: pd.DataFrame({"x": [1]})}
    ensure_tables(tmp_path, builders, None)
    ensure_tables(tmp_path, builders, None)
    messages = [r.getMessage() for r in caplog.records]
    assert any("building" in m and "a" in m for m in messages)
    assert any("wrote" in m and "a.tsv" in m for m in messages)
    assert any("kept existing" in m and "a.tsv" in m for m in messages)
