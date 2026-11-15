# `boldtailor.ridge_selection`

[Inventory](README.md) · [Review routes](../code-review.md)

[Source](../../src/boldtailor/ridge_selection.py)

Select one ridge penalty using held-out trial encoding across features.

## Definitions

- [`_alpha_grid`](#definition-1)
- [`_validate_percentile`](#definition-2)
- [`select_ridge_penalty`](#definition-3)
- [`score_ridge_candidates`](#definition-4)

<a id="definition-1"></a>

## `_alpha_grid`

[Source, line 21](../../src/boldtailor/ridge_selection.py#L21)

```python
_alpha_grid(alphas)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/ridge_selection.py:55](../../src/boldtailor/ridge_selection.py#L55)
- [src/boldtailor/ridge_selection.py:99](../../src/boldtailor/ridge_selection.py#L99)
- [src/boldtailor/workflow/beta_series.py:51](../../src/boldtailor/workflow/beta_series.py#L51)

<a id="definition-2"></a>

## `_validate_percentile`

[Source, line 28](../../src/boldtailor/ridge_selection.py#L28)

```python
_validate_percentile(percentile)
```

No source docstring. Read the linked implementation and call sites for the precise contract.

Usage in package/tests:

- [src/boldtailor/ridge_selection.py:59](../../src/boldtailor/ridge_selection.py#L59)

<a id="definition-3"></a>

## `select_ridge_penalty`

[Source, line 35](../../src/boldtailor/ridge_selection.py#L35)

```python
select_ridge_penalty(scores: CandidateScores | np.ndarray, alphas: Sequence[float] | None=None, *, percentile: float=90.0, feature_mask: np.ndarray | None=None) -> RidgeSelection
```

Reduce once across the complete feature population, never per block.

Pass the ``CandidateScores`` from :func:`score_ridge_candidates`, or a
candidate-by-feature score array together with its ``alphas``.

Only features finite for every candidate enter the common mask. Negative
scores are valid. Objective ties within ``TIE_TOLERANCE`` prefer the smaller
penalty. ``at_boundary`` is True when the winner is the smallest or largest alpha
(always True for a one-point grid).

Usage in package/tests:

- [src/boldtailor/workflow/beta_series.py:219](../../src/boldtailor/workflow/beta_series.py#L219)
- [tests/test_recovery.py:97](../../tests/test_recovery.py#L97)
- [tests/workflow/test_beta_series.py:490](../../tests/workflow/test_beta_series.py#L490)

<a id="definition-4"></a>

## `score_ridge_candidates`

[Source, line 80](../../src/boldtailor/ridge_selection.py#L80)

```python
score_ridge_candidates(data: AnalysisData, predictors: Sequence[pd.DataFrame], *, alphas: Sequence[float], library: HrfLibrary | None=None, run_labels: Sequence[str] | None=None, feature_signature: str | None=None, encoding_mode: str='within_run') -> CandidateScores
```

Score candidate-regularized trial betas with leave-one-run-out encoding.

Supply only outer-training runs when nesting this selection. A library
selects HRFs separately on each inner-training set; None uses canonical
SPM. This returns feature-level scores, without choosing a block-local
penalty. Combine blocks before calling select_ridge_penalty.

Usage in package/tests:

- [tests/test_recovery.py:96](../../tests/test_recovery.py#L96)
- [tests/test_result_schemas.py:165](../../tests/test_result_schemas.py#L165)
- [tests/workflow/test_beta_series.py:489](../../tests/workflow/test_beta_series.py#L489)
