import numpy as np
import pytest
from numpy.testing import assert_allclose

from boldtailor._reml_terms import project_fixed, reml_terms
from boldtailor.ar1 import whiten
from tests.reml_oracles import simulate


@pytest.fixture
def group():
    signals, fixed, random = simulate(
        np.random.default_rng(11), scans=(50,), n_features=3, rho=0.3
    )
    return signals[0], fixed[0], random[0]


def test_energy_decomposes(group):
    y, c, z = group
    terms = reml_terms(project_fixed(y, c, 0.3), z)
    total = terms.residual_ss + terms.coordinates2.sum(0)
    assert_allclose(terms.energy, total, rtol=1e-10)


def test_projected_signals_are_orthogonal_to_fixed(group):
    y, c, _ = group
    projection = project_fixed(y, c, 0.3)
    assert_allclose(whiten(c, 0.3).T @ projection.signals, 0, atol=1e-9)


def test_dof_counts_rank_not_columns(group):
    y, c, _ = group
    duplicated = np.column_stack([c, c[:, 1]])
    assert project_fixed(y, duplicated).dof == project_fixed(y, c).dof


def test_no_restricted_dof_raises():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="positive restricted degrees of freedom"):
        project_fixed(rng.normal(size=(5, 2)), rng.normal(size=(5, 5)))


def test_random_inside_fixed_span_has_no_directions(group):
    y, c, _ = group
    z = c[:, :2] @ np.random.default_rng(1).normal(size=(2, 3))
    assert len(reml_terms(project_fixed(y, c, 0.3), z).singular2) == 0


def test_duplicated_random_column_drops_rank(group):
    y, c, z = group
    duplicated = np.column_stack([z, z[:, 0]])
    terms = reml_terms(project_fixed(y, c), duplicated)
    assert len(terms.singular2) == duplicated.shape[1] - 1


@pytest.mark.parametrize("empty", [None, np.empty((50, 0))])
def test_missing_random_design_has_no_directions(group, empty):
    y, c, _ = group
    terms = reml_terms(project_fixed(y, c), empty)
    assert terms.singular2.shape == (0,)
    assert terms.coordinates2.shape == (0, y.shape[1])
    assert_allclose(terms.residual_ss, terms.energy)


def test_whitening_applied_to_all_inputs(group):
    y, c, _ = group
    direct = project_fixed(y, c, 0.4)
    manual = project_fixed(whiten(y, 0.4), whiten(c, 0.4), 0.0)
    assert_allclose(direct.signals, manual.signals, atol=1e-10)
    assert_allclose(direct.energy, manual.energy, rtol=1e-10)
    assert direct.dof == manual.dof
