"""The two trial-beta penalty families, described once for every caller.

Normalized ridge shares one alpha across features and validates the
candidate-regularized betas. Fractional ridge chooses a fraction per feature
and validates against fixed OLS betas in the held-out run.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from boldtailor._fractional_ridge import (
    NORM_BASIS,
    fraction_grid,
    prepare_fraction_betas,
)
from boldtailor._single_trial_fit import prepare_trial_betas, validate_alpha


def alpha_grid(alphas):
    grid = tuple(validate_alpha(a) for a in alphas)
    if not grid or len(set(grid)) != len(grid):
        raise ValueError("alphas must be a nonempty grid of distinct penalties")
    return grid


@dataclass(frozen=True)
class Regularizer:
    """One penalty family: its grid, its beta path, and how it is validated.

    ``fixed_target`` is the penalty whose betas serve as the held-out target
    for every candidate; None validates each candidate's own betas.
    ``cv_activity`` holds the family's entries in the scoring provenance.
    """

    name: str
    grid_name: str
    fit_option: str
    validation_target: str
    per_feature: bool
    tie_rule: str
    prepare: Callable = field(repr=False)
    parse_grid: Callable = field(repr=False)
    fixed_target: float | None = None
    cv_activity: Mapping[str, object] = field(default_factory=dict, repr=False)

    def grid(self, values) -> tuple[float, ...]:
        """Validated candidates ordered from least to most shrinkage."""
        return tuple(sorted(self.parse_grid(values), reverse=self.per_feature))


NORMALIZED_RIDGE = Regularizer(
    name="normalized_ridge",
    grid_name="alphas",
    fit_option="ridge_alpha",
    validation_target="candidate_regularized_betas",
    per_feature=False,
    tie_rule="smallest alpha within 1e-12 of the maximum objective",
    prepare=prepare_trial_betas,
    parse_grid=alpha_grid,
    cv_activity=MappingProxyType(
        dict(
            name="encoding_guided_ridge_cv",
            normalization="unit_l2_after_nuisance_projection",
        )
    ),
)

FRACTIONAL_RIDGE = Regularizer(
    name="fractional_ridge",
    grid_name="fractions",
    fit_option="ridge_fraction",
    validation_target="fixed_ols_betas",
    per_feature=True,
    tie_rule="largest fraction within 1e-12 of the maximum score",
    prepare=prepare_fraction_betas,
    parse_grid=fraction_grid,
    fixed_target=1.0,
    cv_activity=MappingProxyType(
        dict(
            name="encoding_guided_fractional_ridge_cv",
            normalization="none_after_nuisance_projection",
            fraction_norm_basis=NORM_BASIS,
        )
    ),
)

_BY_NAME = {family.name: family for family in (NORMALIZED_RIDGE, FRACTIONAL_RIDGE)}


def regularizer_named(name) -> Regularizer:
    """The family recorded under ``name`` in scores and fitted records."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(f"unknown regularization: {name!r}") from None
