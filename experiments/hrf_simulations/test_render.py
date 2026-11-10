"""Rendering must use uv's interpreter without installing a global kernel."""

import os
import sys
from types import SimpleNamespace

import nbformat


def test_render_uses_current_interpreter_and_restores_environment(
    tmp_path, monkeypatch
):
    from experiments.hrf_simulations import render

    source = tmp_path / "source.py"
    source.write_text("# %%\nx = 1\n")
    output = tmp_path / "rendered.ipynb"
    report = tmp_path / "report"
    observed = {}

    def client(book, km, timeout):
        observed["argv"] = km.kernel_spec.argv
        observed["report"] = os.environ["HRF_SIM_REPORT"]
        return SimpleNamespace(execute=lambda: book)

    monkeypatch.setattr(render, "NotebookClient", client)
    monkeypatch.setenv("HRF_SIM_REPORT", "previous-value")
    render.render_notebook(source, output, report)
    assert observed["argv"][0] == sys.executable
    assert observed["report"] == str(report.resolve())
    assert os.environ["HRF_SIM_REPORT"] == "previous-value"
    assert nbformat.read(output, as_version=4).cells[0].source == "x = 1"
