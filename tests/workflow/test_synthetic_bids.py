import subprocess
import sys


def test_make_six_runs_does_not_crash():
    code = """
from pathlib import Path
import tempfile
from tests.workflow import synthetic_bids

with tempfile.TemporaryDirectory() as d:
    root, prep, *_ = synthetic_bids.write_dataset(
        Path(d), synthetic_bids.make_confounds(), synthetic_bids.make_events()
    )
    synthetic_bids.make_six_runs(root, prep)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
