# Staged VTC pilot results

Completed local pilots for sub-07, sessions 11 and 12, using all 12 runs per session and 1,750 anatomically defined bilateral fusiform vertices. The experiment scope is sessions 10–19; all 120 requested runs have matched CIFTI, NSD motion, and events files. These results cover two sessions and one complementary pair of 6/6 run splits per session, not the complete ten-session study.

All methods use the confirmed 24-parameter NSD motion model, model-generated 80-second cosine drifts, binary trial type, common RT modulation, and separate missing/incorrect-response junk regressors. Only the HRF is frozen on held-out runs; all amplitudes are refitted.

Values below are medians across vertices after pooling sums of squares across the two complementary test halves. Each session uses the same valid vertices for all three methods (1,749 in session 11; 1,750 in session 12). One session-11 REML half-fit selected the zero-random-variance boundary and therefore did not identify an HRF.

| Session | Method | Total R² | Partial task R² | Incremental task R² |
|---|---|---:|---:|---:|
| 11 | canonical | 0.6175 | 0.0930 | 0.0389 |
| 11 | librarycv | 0.6542 | 0.1482 | 0.0622 |
| 11 | reml | 0.6536 | 0.1449 | 0.0611 |
| 12 | canonical | 0.6522 | 0.1174 | 0.0488 |
| 12 | librarycv | 0.6707 | 0.1358 | 0.0573 |
| 12 | reml | 0.6700 | 0.1352 | 0.0566 |

Both library methods improve task-related variance explained relative to canonical SPM in these pilots. CV and direct REML are close; these two sessions do not establish a preferred estimator. Total R² includes nuisance effects and should not be interpreted as task-related variance.

## HRF stability

Within-session median waveform correlations for CV were 0.972 and 0.963 in sessions 11 and 12; REML correlations were 0.956 and 0.951. Between the two all-run session estimates, median correlations were 0.951 (CV) and 0.947 (REML). Median absolute peak-time differences between sessions were 0.4 s (CV) and 0.5 s (REML). Canonical stability is identically one by construction. These are descriptive summaries, not independent-vertex significance tests.

## Continuous-optimizer check

A separate four-vertex, four-run smoke check completed with both L-BFGS-B and Powell. It used a reduced 32-sample timing library for initialization and a three-value lambda grid, so it is not a fair final performance ranking against the 533-kernel library methods. L-BFGS-B selected lambda 0; Powell selected 0.01. Both produced finite held-out fits. Observed elapsed times including inner tuning were approximately 10 s and 63 s, respectively. Full-VTC nested continuous optimization has not been run.

## Validation and reproducibility

The package suite passed all 1,671 tests. The experiment suite now contains 64 tests, including dense REML objective checks, synthetic off-grid recovery, anatomical identity checks, failure/boundary diagnostics, and verified restart. An independent final review found no substantive blockers for staged local pilots. The optimized CV implementation reproduced the four-run pilot partial R² values exactly (maximum absolute difference 0).

The expanded two-session runs completed before the final CV speed refactor and cache-fingerprint hardening; their original manifests are retained rather than relabeled as new runs. Those final changes did not alter the scientific model. `pilot-session11-final` verifies the numerically equivalent optimized path on the original 256-vertex pilot. A new output directory is required when the implementation fingerprint changes.

Files: [pooled score summary](reports/pilot_summary.tsv), [stability summary](reports/pilot_stability.tsv), [session 11 scores](outputs/vtc-session11/scores.tsv), [session 12 scores](outputs/vtc-session12/scores.tsv), and [execution instructions](README.md). Full maps, HRF curves, per-run sums of squares, diagnostics, manifests, and completion records remain in the output directories.

Next local stage: extend continuous estimation on a modest, anatomically sampled VTC subset using the same initialization library and tuning grid as the library-method comparisons; inspect convergence, boundaries, and temporal residual structure before scaling. Full ten-session/20-split execution and broad simulation sensitivities remain outstanding.
