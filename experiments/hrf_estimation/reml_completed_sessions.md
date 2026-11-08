# Completed-session REML comparison

Completed full-VTC sessions 10 and 11 only; session 12 and 13 outputs are incomplete and excluded. Both sessions used 1,750 anatomical VTC vertices, all 12 runs, the full 533-HRF library, the RT-amplitude model, and one complementary 6/6 partition (two transfer directions). All completion artifact hashes were verified. Only the HRF transfers; amplitudes are refitted on held-out runs. Scoring uses the common unwhitened evaluation model, so this assesses HRF transfer rather than whitened likelihood or inferential calibration.

Original REML has one variance shared by independent task/junk coefficients. AR(1) adds training-only pooled autocorrelation/whitening. Separate components assigns distinct condition, RT and junk variances. Shared effects adds a task component shared across training runs (not shared junk). Combined includes AR(1), separate components and shared condition/RT effects.

## Held-out performance

For each session, use the same vertices with finite scores across all five methods and both transfer directions: 1,716 in session 10 and 1,728 in session 11. Sum SSE and denominators across both held-out halves and vertices before forming R². This weights by residual energy rather than taking a mean of vertexwise R². Partial task R² compares task+junk+nuisance with junk+nuisance; total R² includes all regressors.

| Session | Method | Total R² | Partial task R² | Incremental task R² |
|---|---|---:|---:|---:|
| 10 | reml | 0.686195 | 0.229991 | 0.093729 |
| 10 | remlar1 | 0.685914 | 0.229514 | 0.093561 |
| 10 | remlcomponents | 0.686266 | 0.230197 | 0.093817 |
| 10 | remlshared | 0.686239 | 0.230117 | 0.093783 |
| 10 | remlcombined | 0.686092 | 0.230023 | 0.093777 |
| 11 | reml | 0.709841 | 0.265243 | 0.104746 |
| 11 | remlar1 | 0.709856 | 0.265377 | 0.104813 |
| 11 | remlcomponents | 0.709809 | 0.265207 | 0.104738 |
| 11 | remlshared | 0.709869 | 0.265363 | 0.104800 |
| 11 | remlcombined | 0.710271 | 0.266444 | 0.105236 |

Combined has the largest transfer gain in session 11: +0.001201 partial R² over original REML, or 0.120 percentage points. In session 10 its gain is only +0.000032; separate components leads by +0.000206. AR(1) alone slightly reduces pooled transfer performance in session 10 and slightly improves it in session 11. Thus no consistent, substantial transfer benefit is established. Median vertexwise partial R² is highest for combined in both sessions; the pooled and median summaries answer different weighting questions.

## HRF shape stability

Median Pearson correlation between each vertex’s two independently fitted transfer HRF curves (same common vertex set). This measures split-half shape agreement, not accuracy.

| Session | Method | Median curve correlation | Exact library-ID agreement |
|---|---|---:|---:|
| 10 | reml | 0.9596 | 8.39% |
| 10 | remlar1 | 0.9660 | 11.48% |
| 10 | remlcomponents | 0.9585 | 8.22% |
| 10 | remlshared | 0.9607 | 9.27% |
| 10 | remlcombined | 0.9710 | 13.11% |
| 11 | reml | 0.9565 | 4.75% |
| 11 | remlar1 | 0.9689 | 8.28% |
| 11 | remlcomponents | 0.9564 | 5.21% |
| 11 | remlshared | 0.9583 | 5.15% |
| 11 | remlcombined | 0.9731 | 9.32% |

AR(1) and combined improve descriptive shape agreement in both sessions. Combined has the highest agreement. Similar curves can have different IDs in this dense library, so exact ID agreement is much lower than curve correlation. Greater stability can also reflect shrinkage and is not itself evidence of better HRF recovery.

## Diagnostics and interpretation

Training-only pooled AR coefficients are about 0.52–0.54. Selected extended-model fits overwhelmingly converged, and none hit an upper variance bound. Across the two transfer directions per session, AR(1) has 12/11 nonconverged fits and 35/22 zero-variance fits in sessions 10/11, respectively. Combined has one nonconverged fit per session and one/zero zero-variance fits. Nonconverged fits with finite scores are included in the primary table; restricting all methods/directions to status `ok` leaves 1,694/1,706 common vertices and retains the same performance rankings. See `reports/reml_completed_sessions_converged.tsv`.

These saved AR(1) outputs predate the scalar-solver speed fix. The new solver matched all 1,750 all-run session-11 HRF winners, but transfer fits have not been rerun, so their historical convergence flags remain as recorded.

The evidence supports keeping original and AR(1) as simpler references and treating combined as a stability-oriented sensitivity model; separate/shared alone add little here. Two sessions from one participant and one partition per session do not establish statistical significance or broad generalization. No further multi-session expansion of the complex models is needed for the user’s current scope. All analysis queues remain stopped.

Machine-readable tables: `reports/reml_completed_sessions_summary.tsv`, `reports/reml_completed_sessions_stability.tsv`, `reports/reml_completed_sessions_diagnostics.tsv`, and `reports/reml_completed_sessions_converged.tsv`.
