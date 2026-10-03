"""The recovery report runs every recovery test and prints its quantities."""

import io
from contextlib import redirect_stdout

import recovery_report


def test_report_prints_every_recovery_test_with_its_quantities():
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        statuses = recovery_report.main()
    text = buffer.getvalue()
    assert set(statuses.values()) == {"PASS"}
    for name in recovery_report.TESTS:
        assert f"## {name.__name__}: PASS" in text
    assert "centered RMSE" in text and "selected alpha" in text
    assert "selected ids" in text and "canonical fraction" in text
