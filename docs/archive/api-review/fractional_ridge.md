# `boldtailor.fractional_ridge`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/fractional_ridge.py)

Select relative shrinkage independently for each feature using encoding CV.

## Definitions

- [`select_ridge_fractions`](#definition-1)
- [`_choose_fractions`](#definition-2)
- [`score_fraction_candidates`](#definition-3)

<a id="definition-1"></a>

## `select_ridge_fractions`

[Source, line 25](../../src/boldtailor/fractional_ridge.py#L25)

```python
select_ridge_fractions(scores: CandidateScores | np.ndarray, fractions: Sequence[float] | None=None, *, feature_mask: np.ndarray | None=None) -> FractionSelection
```

Maximize each feature's score; ties favor less shrinkage (larger f).

Pass the ``CandidateScores`` from :func:`score_fraction_candidates`, or a
candidate-by-feature score array together with its ``fractions``.

A feature must have finite scores at every candidate. Ineligible features
return NaN fractions/scores and index -1, including entirely invalid blocks.
``at_boundary`` is True for scored features whose winner is the largest or
smallest fraction; with a one-point grid every scored feature is flagged.

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:217](../../src/boldtailor/workflow/beta_series.py#L217)
- [tests/test_recovery.py:77](../../tests/test_recovery.py#L77)
- [tests/test_recovery.py:88](../../tests/test_recovery.py#L88)
- [tests/workflow/test_beta_series.py:265](../../tests/workflow/test_beta_series.py#L265)

<a id="definition-2"></a>

## `_choose_fractions`

[Source, line 54](../../src/boldtailor/fractional_ridge.py#L54)

```python
_choose_fractions(grid, scores, mask)
```

Per-feature winners over a descending grid; ties keep the larger f.

Usage in package/tests:

- [src/boldtailor/fractional_ridge.py:51](../../src/boldtailor/fractional_ridge.py#L51)

<a id="definition-3"></a>

## `score_fraction_candidates`

[Source, line 77](../../src/boldtailor/fractional_ridge.py#L77)

```python
score_fraction_candidates(data: AnalysisData, predictors: Sequence[pd.DataFrame], *, fractions: Sequence[float], library: HrfLibrary | None=None, run_labels: Sequence[str] | None=None, feature_signature: str | None=None, encoding_mode: str='within_run') -> CandidateScores
```

Score fixed OLS beta targets in held-out runs; no image repeats needed.

A supplied library selects HRFs only on each fold's training runs. Pass
outer-training runs only when nesting this scorer. Fractions act on the
raw trial coefficients after nuisance projection; nuisance parameters remain
unpenalized. Only training betas use each candidate fraction; held-out OLS
targets use the same training-selected HRF for all candidates.

Usage in package/tests:

- [tests/test_recovery.py:76](../../tests/test_recovery.py#L76)
- [tests/test_recovery.py:87](../../tests/test_recovery.py#L87)
- [tests/workflow/test_beta_series.py:259](../../tests/workflow/test_beta_series.py#L259)
- [tests/workflow/test_beta_series.py:487](../../tests/workflow/test_beta_series.py#L487)
