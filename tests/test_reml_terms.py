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
    """Verify energy decomposes."""
    y, c, z = group
    terms = reml_terms(project_fixed(y, c, 0.3), z)
    total = terms.residual_ss + terms.coordinates2.sum(0)
    assert_allclose(terms.energy, total, rtol=1e-10)


def test_projected_signals_are_orthogonal_to_fixed(group):
    """Verify projected signals are orthogonal to fixed."""
    y, c, _ = group
    projection = project_fixed(y, c, 0.3)
    assert_allclose(whiten(c, 0.3).T @ projection.signals, 0, atol=1e-9)


def test_dof_counts_rank_not_columns(group):
    """Verify degrees-of-freedom counts rank not columns."""
    y, c, _ = group
    duplicated = np.column_stack([c, c[:, 1]])
    assert project_fixed(y, duplicated).dof == project_fixed(y, c).dof


def test_no_restricted_dof_raises():
    """Verify no restricted degrees-of-freedom raises."""
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="positive restricted degrees of freedom"):
        project_fixed(rng.normal(size=(5, 2)), rng.normal(size=(5, 5)))


def test_random_inside_fixed_span_has_no_directions(group):
    """Verify random inside fixed span has no directions."""
    y, c, _ = group
    z = c[:, :2] @ np.random.default_rng(1).normal(size=(2, 3))
    assert len(reml_terms(project_fixed(y, c, 0.3), z).singular2) == 0


def test_duplicated_random_column_drops_rank(group):
    """Verify duplicated random column drops rank."""
    y, c, z = group
    duplicated = np.column_stack([z, z[:, 0]])
    terms = reml_terms(project_fixed(y, c), duplicated)
    assert len(terms.singular2) == duplicated.shape[1] - 1


@pytest.mark.parametrize("empty", [None, np.empty((50, 0))])
def test_missing_random_design_has_no_directions(group, empty):
    """Verify missing random design has no directions."""
    y, c, _ = group
    terms = reml_terms(project_fixed(y, c), empty)
    assert terms.singular2.shape == (0,)
    assert terms.coordinates2.shape == (0, y.shape[1])
    assert_allclose(terms.residual_ss, terms.energy)


def test_whitening_applied_to_all_inputs(group):
    """Verify whitening applied to all inputs."""
    y, c, _ = group
    direct = project_fixed(y, c, 0.4)
    manual = project_fixed(whiten(y, 0.4), whiten(c, 0.4), 0.0)
    assert_allclose(direct.signals, manual.signals, atol=1e-10)
    assert_allclose(direct.energy, manual.energy, rtol=1e-10)
    assert direct.dof == manual.dof


SHAPES = {"tall": (40, 3), "square": (12, 12), "wide": (6, 90)}


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_spectral_norm_matches_svd(shape):
    """Verify spectral norm matches SVD."""
    from boldtailor._reml_terms import _spectral_norm

    matrix = np.random.default_rng(0).normal(size=shape)
    expected = np.linalg.svd(matrix, compute_uv=False)[0]
    assert _spectral_norm(matrix) == pytest.approx(expected, rel=1e-12)


def test_spectral_norm_of_rank_deficient_and_zero_matrices():
    """Verify spectral norm of rank deficient and zero matrices."""
    from boldtailor._reml_terms import _spectral_norm

    column = np.random.default_rng(1).normal(size=(8, 1))
    repeated = np.tile(column, (1, 30))
    expected = np.linalg.svd(repeated, compute_uv=False)[0]
    assert _spectral_norm(repeated) == pytest.approx(expected, rel=1e-12)
    assert _spectral_norm(np.zeros((4, 9))) == 0.0


@pytest.mark.parametrize("shape", SHAPES.values(), ids=SHAPES.keys())
def test_spectral_norm_decomposes_the_smaller_gram_matrix(shape, monkeypatch):
    """Verify spectral norm decomposes the smaller gram matrix."""
    from boldtailor._reml_terms import _spectral_norm

    seen = []
    eigvalsh = np.linalg.eigvalsh
    monkeypatch.setattr(
        np.linalg, "eigvalsh", lambda a: seen.append(a.shape) or eigvalsh(a)
    )
    _spectral_norm(np.random.default_rng(0).normal(size=shape))
    assert seen == [(min(shape), min(shape))]


def test_wide_random_design_keeps_fixed_span_rank_policy(group):
    """Columns inside the fixed span are still rejected against the
    unprojected norm when the design is wider than it is tall."""
    signals, fixed, _ = group
    rng = np.random.default_rng(2)
    inside = fixed @ rng.normal(size=(fixed.shape[1], 80))
    terms = reml_terms(project_fixed(signals, fixed), inside)
    assert terms.singular2.size == 0


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_spectral_norm_does_not_overflow_or_underflow(scale):
    """Verify spectral norm does not overflow or underflow."""
    from boldtailor._reml_terms import _spectral_norm

    base = np.random.default_rng(0).normal(size=(20, 5))
    expected = scale * np.linalg.svd(base, compute_uv=False)[0]
    assert_allclose(_spectral_norm(scale * base), expected, rtol=1e-12)
