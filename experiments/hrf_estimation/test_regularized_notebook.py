"""The pilot notebook must contain both sessions, all methods, and executed plots."""
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd


def test_regularized_pilot_notebook_is_executed_and_quiet():
    path = Path(__file__).with_name("regularized_ar1_results.ipynb")
    assert path.exists(), "regularized AR1 notebook is missing"
    notebook = nbformat.read(path,as_version=4)
    nbformat.validate(notebook)
    codes = [c for c in notebook.cells if c.cell_type == "code"]
    assert all(isinstance(c.execution_count,int) for c in codes)
    outputs = [o for c in codes for o in c.outputs]
    assert not any(o.output_type == "error" for o in outputs)
    assert sum("image/png" in o.get("data",{}) for o in outputs) >= 5
    assert all(len(o.get("data",{}).get("text/plain","")) < 2000 for o in outputs)


def test_regularized_summary_uses_same_vertices_and_pooled_sse():
    root = Path(__file__).with_name("reports") / "regularized_ar1"
    assert (root/"summary.tsv").exists(), "pilot summary is missing"
    summary = pd.read_csv(root/"summary.tsv",sep="\t")
    scores = pd.read_csv(root/"scores.tsv.gz",sep="\t")
    assert len(summary)==10
    assert set(summary.session)=={11,16}
    assert set(summary.method)=={"pooled","shrinkage","surface","Original REML","Library CV"}
    assert summary.n_common.nunique()==1 and summary.n_common.min()>0
    for row in summary.itertuples():
        data=scores[(scores.method==row.method)&(scores.session==row.session)]
        assert len(data)==2*row.n_common
        assert np.isclose(row.r2_task_partial,1-data.sse_full.sum()/data.sse_nuisance.sum())
        assert -1 <= row.median_shape_r <= 1
