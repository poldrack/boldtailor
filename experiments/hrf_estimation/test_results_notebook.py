"""The deliverable must contain executed results and readable rendered figures."""

import base64
from io import BytesIO
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from PIL import Image
import pytest


@pytest.fixture(scope="module")
def results_notebook(executed_notebook):
    return executed_notebook("results")

def test_notebook_is_fully_executed_without_errors(results_notebook):
    cells = [cell for cell in results_notebook.cells if cell.cell_type == "code"]
    assert cells
    # A full rerun in an existing kernel need not restart its counter at one.
    counts = [cell.execution_count for cell in cells]
    assert all(isinstance(count, int) and count >= 1 for count in counts)
    assert counts == list(range(counts[0], counts[0] + len(cells)))
    assert not [output for cell in cells for output in cell.outputs
                if output.output_type == "error"]


def test_notebook_summary_matches_saved_pooled_results(results_notebook):
    bundles = [output.get("data", {}).get("application/json", {})
               for cell in results_notebook.cells if cell.cell_type == "code"
               for output in cell.outputs]
    summaries = [bundle for bundle in bundles if bundle.get("kind") == "hrf_pooled_summary"]
    assert len(summaries) == 1
    actual = pd.DataFrame(summaries[0]["records"]).sort_values(["session", "method"])
    expected = pd.read_csv(Path(__file__).parent / "reports/pilot_summary.tsv", sep="\t")
    expected = expected.sort_values(["session", "method"])
    assert actual[["session", "method", "n_common"]].to_dict("records") == expected[
        ["session", "method", "n_common"]].to_dict("records")
    metrics = ["r2_full", "r2_task_partial", "delta_r2_task"]
    np.testing.assert_allclose(actual[metrics], expected[metrics], rtol=1e-12, atol=1e-12)


def test_notebook_contains_rendered_result_figures(results_notebook):
    figures = [output.get("data", {}).get("image/png")
               for cell in results_notebook.cells if cell.cell_type == "code"
               for output in cell.outputs if "image/png" in output.get("data", {})]
    assert len(figures) >= 6, "score, shape, stability, and optimizer figures must be rendered"
    for data in figures:
        with Image.open(BytesIO(base64.b64decode(data))) as picture:
            assert picture.width >= 600 and picture.height >= 300
            pixels = np.asarray(picture.convert("RGB"))
            assert pixels.std() > 5, "a result figure is blank"


def test_notebook_includes_verified_reml_validation(results_notebook):
    bundles = [o.get("data", {}).get("application/json", {})
               for c in results_notebook.cells if c.cell_type == "code" for o in c.outputs]
    summaries = [b for b in bundles if b.get("kind") == "reml_validation_summary"]
    assert len(summaries) == 1
    actual = pd.DataFrame(summaries[0]["records"]).sort_values(["session", "method"])
    expected = pd.read_csv(Path(__file__).parent / "reports/reml_validation_summary.tsv", sep="\t")
    expected = expected.sort_values(["session", "method"])
    # Notebook JSON serialization sorts object keys; table column order is separate.
    assert set(actual.columns) == set(expected.columns)
    actual = actual[expected.columns]
    pd.testing.assert_frame_equal(actual.reset_index(drop=True), expected.reset_index(drop=True),
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    text = "\n".join(c.source for c in results_notebook.cells if c.cell_type == "markdown")
    assert "four-vertex REML validation" in text
    figures = [o for c in results_notebook.cells if c.cell_type == "code"
               for o in c.outputs if "image/png" in o.get("data", {})]
    assert len(figures) >= 13
