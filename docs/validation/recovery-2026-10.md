# Method recovery validation (2026-10)

> **Dated validation record:** Results and test counts apply to the recorded
> revision/settings, not necessarily this checkout. See the
> [documentation index](../README.md) for current API and methods guidance.

These results come from `tests/test_recovery.py`, which simulates data with
known truth (AR(1) noise) and checks that the selection methods recover it.
Reproduce with `uv run pytest -q tests/test_recovery.py`.

## Fraction selection (shrinkage)

- Shrinkage case: the selected fraction was 0.4 for both features. Centered
  RMSE against the true betas was 0.81 with ridge versus 2.21 with OLS.
- OLS control: fraction 1.0 was selected. Median CV R² was 0.752 at f=1.0
  versus 0.751 at f=0.8. This margin is thin; it held on seeds 1-3 and 12.

## Ridge penalty selection

- Selected alpha was 10.0, the top of the grid `(0, 0.3, 1, 3, 10)`;
  `at_boundary` is True. Median CV R² was still rising (0.1529 to 0.1531
  between the last two grid points) while alpha=0 scores 0.049.
- The test therefore rejects OLS but does not locate the penalty. The grid
  should be extended in the P1 estimand follow-up.

## HRF selection

- Before the library correction: 6/8 features recovered. Both misses were
  canonical-truth features confused with library id 3, a custom kernel
  `[5, 14, 1.0, 1.5, 6, 1.0, 36]` whose cosine similarity to the canonical
  kernel is 0.995, so the test passed at exactly its 0.75 threshold.
- After replacing id 3 with the well-separated kernel
  `[4, 10, 0.6, 0.8, 3, 0.3, 36]` (cosine to canonical 0.851 when both are
  sampled with `kernel(1.6, 50)` and zero-padded): selected ids
  `[0, 1, 2, 3, 2, 1, 0, 3]` equal the truth, i.e. 8/8 = 1.0 recovered
  (threshold 0.75, unchanged). The earlier misses were therefore caused by the
  near-duplicate library entry, not by a selection bias.
- Canonical control (truth canonical for all 6 features; library now includes
  the near-canonical `[5, 14, 1.0, 1.5, 6, 1.0, 36]` row, cosine 0.995 to
  canonical): selected ids `[2, 0, 0, 0, 0, 0]`, i.e. 5/6 = 0.833 selected
  canonical (threshold 0.75). The one miss chose the widest library kernel
  (`[6, 16, 1.5, 2.5, 8, 2, 36]`), not the near tie. The margin over the
  threshold is a single feature.
