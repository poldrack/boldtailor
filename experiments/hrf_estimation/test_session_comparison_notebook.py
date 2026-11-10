"""Focused notebook: complete ten-session methods, then complete REML variants."""
from pathlib import Path
import nbformat
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def notebook(executed_notebook):
    return executed_notebook("session_comparison")

@pytest.fixture(scope="module")
def notebook_verbose(executed_notebook):
    return executed_notebook("session_comparison", long_outputs=True)


def records(notebook, kind):
    bundles=[o.get("data",{}).get("application/json",{})
             for c in notebook.cells if c.cell_type=="code" for o in c.outputs]
    return next(b["records"] for b in bundles if b.get("kind")==kind)


def test_only_complete_ten_session_methods(notebook_verbose):
    table=pd.DataFrame(records(notebook_verbose,"ten_session_comparison"))
    assert set(table.method)=={"Canonical","Library CV","Parametric library CV","Original REML","AR(1) REML"}
    assert len(table)==50
    for _,group in table.groupby('method'):
        assert set(group.session)==set(range(10,20))
    assert table.n_common.nunique()==1


def test_only_complete_two_session_reml_variants(notebook_verbose):
    table=pd.DataFrame(records(notebook_verbose,"reml_session_comparison"))
    assert set(table.method)=={"Original REML","AR(1) REML","Separate components","Shared effects","Combined"}
    assert len(table)==10 and set(table.session)=={10,11}


def test_focused_notebook_executes_with_figures(notebook):
    cells=[c for c in notebook.cells if c.cell_type=='code']
    assert cells and all(isinstance(c.execution_count,int) for c in cells)
    outputs=[o for c in cells for o in c.outputs]
    assert not any(o.output_type=='error' for o in outputs)
    assert sum('image/png' in o.get('data',{}) for o in outputs)>=4


def test_shape_comparisons_cover_all_core_methods_and_sessions(notebook_verbose):
    within=pd.DataFrame(records(notebook_verbose,"core_shape_stability"))
    expected={"Canonical","Library CV","Parametric library CV","Original REML","AR(1) REML"}
    assert set(within.method)==expected and len(within)==50
    for _,g in within.groupby('method'):
        assert set(g.session)==set(range(10,20))
    pairs=pd.DataFrame(records(notebook_verbose,"cross_session_shapes"))
    assert set(pairs.method)==expected and len(pairs)==500
    canonical=pairs[pairs.method.eq('Canonical')]
    assert (canonical.median_curve_r-1).abs().max()<1e-12
    assert pairs.n_common.nunique()==1


def test_empirical_selection_report_has_counts_and_session_agreement(notebook_verbose):
    rates=pd.DataFrame(records(notebook_verbose,"empirical_selection_rates"))
    assert len(rates)==20
    assert set(rates.session)==set(range(10,20))
    assert (rates.n_empirical+rates.n_parametric==rates.n_common).all()
    assert rates.n_common.nunique()==1
    assert rates.query("method == 'Parametric library CV'").n_empirical.eq(0).all()
    assert rates.query("method == 'Library CV'").n_empirical.gt(0).any()
    pairs=pd.DataFrame(records(notebook_verbose,"empirical_selection_agreement"))
    assert len(pairs)==45
    assert (pairs.session_a<pairs.session_b).all()
    assert (pairs.n_both_empirical+pairs.n_both_parametric+pairs.n_switched==pairs.n_common).all()
    assert pairs.family_agreement.between(0,1).all()
    assert pairs.empirical_jaccard.between(0,1).all()
    assert (pairs.n_same_empirical_id<=pairs.n_both_empirical).all()


def test_long_outputs_disabled_by_default(notebook):
    setup = next(c.source for c in notebook.cells if c.cell_type == "code")
    assert "no_long_text_outputs = True" in setup
    outputs = [o for c in notebook.cells if c.cell_type == "code" for o in c.outputs]
    assert not any("application/json" in o.get("data", {}) for o in outputs)
    assert all(len(o.get("data", {}).get("text/plain", "")) < 2000 for o in outputs)
    assert sum("image/png" in o.get("data", {}) for o in outputs) >= 11
    namespace = {"display": lambda *args, **kwargs: calls.append(args)}
    helper = setup[setup.index("def display_long_output"):setup.index("root=next")]
    exec(helper, namespace)
    calls = []
    namespace["no_long_text_outputs"] = True
    namespace["display_long_output"]("long output")
    assert calls == []
    namespace["no_long_text_outputs"] = False
    namespace["display_long_output"]("long output")
    assert calls == [("long output",)]


def test_rt_ablation_covers_all_sessions_and_crossed_models(notebook_verbose):
    table=pd.DataFrame(records(notebook_verbose,"rt_ablation_comparison"))
    assert len(table)==40
    assert set(table.session)==set(range(10,20))
    assert set(zip(table.selection_rt,table.evaluation_rt))=={(a,b) for a in ["amplitude","none"] for b in ["amplitude","none"]}
    assert table.n_common.nunique()==1
    for _,g in table.groupby(["session","selection_rt"]):
        g=g.set_index("evaluation_rt")
        assert g.loc["amplitude","r2_full"] >= g.loc["none","r2_full"]-1e-12
    shape=pd.DataFrame(records(notebook_verbose,"rt_ablation_shapes"))
    assert len(shape)==20 and set(shape.selection_rt)=={"amplitude","none"}
    assert shape.n_common.nunique()==1
    assert shape.median_curve_r.between(-1,1).all()
