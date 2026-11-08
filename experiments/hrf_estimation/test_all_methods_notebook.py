"""The comparison notebook must display verified coverage and pooled results."""
from pathlib import Path
import nbformat
import pandas as pd
import pytest


@pytest.fixture
def notebook():
    path=Path(__file__).with_name("all_methods_results.ipynb")
    assert path.exists(), "comparison notebook is missing"
    result=nbformat.read(path,as_version=4)
    nbformat.validate(result)
    return result


def records(notebook, kind):
    bundles=[o.get("data",{}).get("application/json",{})
             for c in notebook.cells if c.cell_type=="code" for o in c.outputs]
    return next(b["records"] for b in bundles if b.get("kind")==kind)


def test_notebook_executes_and_renders_comparisons(notebook):
    cells=[c for c in notebook.cells if c.cell_type=="code"]
    assert cells and all(isinstance(c.execution_count,int) for c in cells)
    outputs=[o for c in cells for o in c.outputs]
    assert not any(o.output_type=="error" for o in outputs)
    assert sum("image/png" in o.get("data",{}) for o in outputs)>=4


def test_coverage_distinguishes_missing_models(notebook):
    coverage=pd.DataFrame(records(notebook,"model_coverage"))
    ar1=coverage.query("method == 'AR(1) REML' and mode == 'amplitude' and state == 'complete'")
    assert set(ar1.session)==set(range(10,20))
    missing=coverage.query("method == 'Continuous L-BFGS-B' and mode == 'duration' and session == 12")
    assert len(missing)==1 and missing.iloc[0].state!='complete'


def test_session14_ar1_score_matches_raw_output(notebook):
    summary=pd.DataFrame(records(notebook,"all_methods_summary"))
    row=summary.query("session == 14 and mode == 'amplitude' and method == 'AR(1) REML'").iloc[0]
    source=Path(__file__).parent/'outputs/full-vtc-ar1-scalar/session14-amplitude/scores.tsv'
    data=pd.read_csv(source,sep="\t")
    transfer=data[data.evaluation.str.startswith('transfer')]
    valid=set.intersection(*[set(g.loc[g.r2_task_partial.notna(),'feature_id'])
                            for _,g in transfer.groupby('evaluation')])
    transfer=transfer[transfer.feature_id.isin(valid)]
    assert row.n_common==len(valid)
    expected=1-transfer.sse_full.sum()/transfer.sse_nuisance.sum()
    assert row.r2_task_partial==pytest.approx(expected,abs=1e-12)


def test_prior_modeling_approaches_are_included(notebook):
    coverage=pd.DataFrame(records(notebook,"model_coverage"))
    available=set(coverage.loc[coverage.state.eq("complete"),"method"])
    assert {"Canonical","Library CV","Original REML","Parametric library CV",
            "Parametric REML","Continuous L-BFGS-B"} <= available
    historical=pd.DataFrame(records(notebook,"historical_models"))
    assert "Powell" in set(historical.method)


def test_baseline_comparators_cover_all_ar1_sessions(notebook):
    coverage=pd.DataFrame(records(notebook,"model_coverage"))
    for method in ["Canonical","Library CV","Parametric library CV","Original REML","AR(1) REML"]:
        completed=coverage.query("mode == 'amplitude' and state == 'complete' and method == @method")
        assert set(completed.session)==set(range(10,20)), method
