import numpy as np
import pytest

from experiments.nsd_replication.confounds import glmsingle_polynomials, polynomial_degree


@pytest.mark.parametrize(
    "n, tr, degree", [(226, 4 / 3, 3), (226, 1.6, 3), (75, 1.6, 1), (450, 1.0, 4)]
)
def test_degree_rule(n, tr, degree):
    assert polynomial_degree(n, tr) == degree


def test_degree_rounds_half_away_from_zero():
    assert polynomial_degree(150, 1.0) == 1  # 1.25 -> 1
    assert polynomial_degree(180, 1.0) == 2  # 1.5 -> 2
    assert polynomial_degree(300, 1.0) == 3  # 2.5 -> 3 (Python round gives 2)


def test_basis_is_orthonormal_and_spans_monomials():
    basis = glmsingle_polynomials(226, 4 / 3).to_numpy()
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-10)
    x = np.linspace(-1, 1, 226)
    monomials = np.column_stack([x**d for d in range(4)])
    projection = basis @ basis.T
    np.testing.assert_allclose(projection @ monomials, monomials, atol=1e-8)
    assert list(glmsingle_polynomials(226, 4 / 3).columns) == [
        "poly_0",
        "poly_1",
        "poly_2",
        "poly_3",
    ]
