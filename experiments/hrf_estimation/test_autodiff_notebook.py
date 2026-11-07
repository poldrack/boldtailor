"""Verify the executed autodiff report against the saved benchmark."""
from pathlib import Path

import nbformat
import pandas as pd
import pytest


@pytest.fixture
def notebook():
    path = Path(__file__).with_name("autodiff_results.ipynb")
    assert path.exists(), "autodiff results notebook is missing"
    result = nbformat.read(path, as_version=4)
    nbformat.validate(result)
    return result


def test_notebook_executed_with_figures(notebook):
    cells = [c for c in notebook.cells if c.cell_type == "code"]
    assert cells and all(isinstance(c.execution_count, int) for c in cells)
    outputs = [o for c in cells for o in c.outputs]
    assert not any(o.output_type == "error" for o in outputs)
    assert sum("image/png" in o.get("data", {}) for o in outputs) >= 3


def test_notebook_preserves_benchmark_records(notebook):
    bundles = [o.get("data", {}).get("application/json", {})
               for c in notebook.cells if c.cell_type == "code" for o in c.outputs]
    records = [b["records"] for b in bundles if b.get("kind") == "autodiff_benchmark"]
    assert len(records) == 1
    actual = pd.DataFrame(records[0])
    expected = pd.read_csv(Path(__file__).parent / "reports/autodiff_benchmark_session11.tsv", sep="\t")
    pd.testing.assert_frame_equal(actual[expected.columns], expected, rtol=1e-12, atol=1e-12)
