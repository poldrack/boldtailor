"""Execute notebook sources afresh; never trust saved outputs in local ipynb files."""
from functools import lru_cache
from pathlib import Path
import sys

import jupytext
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
import pytest


@pytest.fixture(scope="session")
def executed_notebook():
    @lru_cache(maxsize=None)
    def execute(name, long_outputs=False):
        path = Path(__file__).with_name(name + ".py")
        assert path.exists(), f"Python notebook source is missing: {path.name}"
        notebook = jupytext.read(path)
        if long_outputs:
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    cell.source = cell.source.replace("no_long_text_outputs = True", "no_long_text_outputs = False")
        kernel = KernelManager()
        kernel.kernel_spec.argv = [sys.executable,"-m","ipykernel_launcher","-f","{connection_file}"]
        client = NotebookClient(notebook, timeout=180, km=kernel,
                                resources={"metadata":{"path":str(path.parent)}})
        client.execute()
        nbformat.validate(notebook)
        return notebook
    return execute
