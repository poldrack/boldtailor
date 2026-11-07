"""The deliverable must contain executed results and readable rendered figures."""

import base64
from io import BytesIO
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from PIL import Image
import pytest


@pytest.fixture
def results_notebook():
    path = Path(__file__).with_name("results.ipynb")
    assert path.is_file(), "the results notebook has not been created"
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    return notebook


def test_notebook_is_fully_executed_without_errors(results_notebook):
    cells = [cell for cell in results_notebook.cells if cell.cell_type == "code"]
    assert cells
    assert [cell.execution_count for cell in cells] == list(range(1, len(cells) + 1))
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
