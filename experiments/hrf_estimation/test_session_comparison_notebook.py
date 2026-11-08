"""Focused notebook: complete ten-session methods, then complete REML variants."""
from pathlib import Path
import nbformat
import pandas as pd
import pytest


@pytest.fixture
def notebook():
    path=Path(__file__).with_name("session_comparison.ipynb")
    assert path.exists(), "focused session comparison notebook is missing"
    result=nbformat.read(path,as_version=4)
    nbformat.validate(result)
    return result


def records(notebook, kind):
    bundles=[o.get("data",{}).get("application/json",{})
             for c in notebook.cells if c.cell_type=="code" for o in c.outputs]
    return next(b["records"] for b in bundles if b.get("kind")==kind)


def test_only_complete_ten_session_methods(notebook):
    table=pd.DataFrame(records(notebook,"ten_session_comparison"))
    assert set(table.method)=={"Canonical","Library CV","Parametric library CV","Original REML","AR(1) REML"}
    assert len(table)==50
    for _,group in table.groupby('method'):
        assert set(group.session)==set(range(10,20))
    assert table.n_common.nunique()==1


def test_only_complete_two_session_reml_variants(notebook):
    table=pd.DataFrame(records(notebook,"reml_session_comparison"))
    assert set(table.method)=={"Original REML","AR(1) REML","Separate components","Shared effects","Combined"}
    assert len(table)==10 and set(table.session)=={10,11}


def test_focused_notebook_executes_with_figures(notebook):
    cells=[c for c in notebook.cells if c.cell_type=='code']
    assert cells and all(isinstance(c.execution_count,int) for c in cells)
    outputs=[o for c in cells for o in c.outputs]
    assert not any(o.output_type=='error' for o in outputs)
    assert sum('image/png' in o.get('data',{}) for o in outputs)>=4
