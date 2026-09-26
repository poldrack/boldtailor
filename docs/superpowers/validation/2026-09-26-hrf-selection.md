# Expanded HRF selection validation — 2026-09-26

The approved mean-stimulus selection design is implemented. The full
sub-07/ses-nsd10 analysis completed and its saved results passed an independent
numerical audit. Code was integrated by local fast-forward into main, preserving
the user's existing stop-signal notebook and test edits. No remote push occurred.

## Analysis and audit

- Input: `/Volumes/extdata1/NSD/BIDS/sub-07/ses-nsd10`, with fMRIPrep 25.2.5
  CIFTIs; 12 runs, 750 trials, 91,282 grayordinates, TR 1.6 s.
- Output: `/Volumes/extdata1/NSD/BIDS/derivatives/boldtailor/sub-07/ses-nsd10/func`.
  Descriptors: `hrfOptOLS`, `hrfOptRidge`, `hrfSelection`.
- Library: 648 notebook combinations plus exact Nilearn SPM candidate 0.
  Notebook SHA256:
  `6e116a740645988332d997ce53dc682977c832a1d8dd3ce82e226d364fa96208`.
  All 648 sampled custom kernels agreed with the notebook implementation to
  maximum absolute error `2.64e-16`.
- All-run leave-one-run-out selection; odd-run inner CV plus frozen odd-trained
  mean response for independent even-run prediction. Neither RT nor repeated
  image identities enter HRF selection. Fixed ridge alpha is 0.1.
- Full run at commit `67f959b`: 720.536 seconds, 6,388,727,808-byte peak RSS;
  72 published files; SHA256 hashes of all 87 existing files preserved.
- Later review fixes at `db1aeac` reject inconsistent direct SPM construction
  and tolerate an unavailable canonical RT diagnostic. Neither condition occurs
  in this dataset, so those changes do not alter this run's numerical results.

The audit selected 25 evenly spaced grayordinates, the five diagnostic vertices,
and two undefined locations (32 distinct indices). It independently convolved
all 649 candidates, projected nuisances with NumPy least squares, fitted each
training mean coefficient, and explicitly calculated held-out prediction errors.
It independently reconstructed grouped trial designs and ordinary/augmented
least-squares fits. Saved axes, trial identities, scores, betas, pooled R², RT
vertex rankings and even-run correlations passed their comparisons. Maximum
absolute beta difference was `3.80e-6` and R² difference `2.97e-8`, consistent
with float32 output. The complete numeric report is in
[the audit JSON](2026-09-26-hrf-selection-audit.json).

Independent even-run median R² was 0.002465 (selected) versus 0.001078
(canonical); median paired ΔR² was 0.000924, positive at 57.93% of defined
grayordinates. The five independent RT correlations ranged from −0.102 to
0.229 for OLS and −0.107 to 0.251 for ridge. Three strengthened and two weakened
relative to the same vertices' canonical estimates. No library or penalty was
retuned using these results. Full numerical summaries are in the example README.

The saved library, selected-HRF and even-run scatter plots were inspected.
Labels and legends are legible; the scatter retains the sparse long-RT tail.

## Engineering verification

Every feature stage followed RED, committed failing tests, then GREEN.
The final review pass added three failing cases before fixing either behavior.
Library and NSD regression tests passed (27 tests). The isolated worktree suite
had 461 passes and the same three pre-existing stop-signal notebook failures
seen before feature implementation. The user's existing main-checkout edits
address those failures and were preserved byte-for-byte during integration.
Final main-checkout command `uv run pytest tests examples/NSD -q -W error
--tb=short` passed **464 tests in 25.52 seconds**, including those three
notebook cases. Main-checkout Black verification also passed for all 54 files;
all source initializers remain empty.

All 54 source/test/example Python files passed Black, and `git diff --check`
passed. Wheel and source distribution builds included the new modules and
retained empty initializers. A fresh independent reviewer assessed the full
feature diff once; the reviewer could not execute numerical reproductions,
so the numerical audit described above was performed by the implementing agent
using an independent calculation path.

Profiling justified optimizing custom trial convolution. For a synthetic
12-run, 32-feature benchmark, eligibility fell from 43.89 to 2.14 seconds,
OLS from 25.45 to 0.95 seconds, and ridge from 42.97 to 0.74 seconds, with the
same 27 selected IDs. A real 4096-feature block took 29.42 seconds for initial
selection, 2.06 seconds for independent evaluation, 17.27 seconds for OLS,
and 12.04 seconds for ridge (2.50 GB peak RSS). The full-run settings remained
4096 features and 32 candidates per batch.

## Review decisions and limitations

One fresh review found no Critical issue. The canonical-ineligible RT abort
was fixed. The direct-SPM metadata issue was graded Important by the executor
because it could silently misidentify a kernel; inconsistent construction now
raises an error. The following decisions record the reasoning and consequences:

| Decision | Reason | Consequence or remaining limitation |
| --- | --- | --- |
| Use an isolated temporary worktree, then locally fast-forward main | Preserve unrelated edits and complete authorized reversible integration | Integration can be reverted locally; no remote changes |
| Use cumulative kernel sums for custom boxcar convolution | Profiling showed trial convolution dominated runtime | Tiny rounding changes are possible; direct-convolution tests enforce rtol `1e-11`, atol `2e-14`; canonical path remains exact |
| Begin final review while the external drive was unavailable | Review code while awaiting real-data access | No real-data success claim until the separate audit passed |
| Reject inconsistent SPM parameter metadata | Public identity must match its fixed canonical kernel | Previously accepted mislabeled candidate objects now raise an error |
| Retain enclosing evaluation provenance | It hashes both training and test designs, including eligibility context | An extracted training-selection record alone lacks complete test-design context |
| Retain existing nuisance rank tolerance | No demonstrated new regression; the feature reuses the established solver | Extreme nuisance rescaling can affect numerical rank decisions |
| Keep defensive public accessors, without hardening private containers | Matches existing ownership conventions | Deliberately mutating private internals remains unsupported |
| Complete one fresh static review and an independent author-run numerical audit | Follows the single-review execution workflow | Numerical verification is independent of the solver but not its author |

One **Minor** archive enhancement remains deferred: the per-run NPZ contains
HRFs used by all-run production fits. An HRF used only by independent RT fits
may be absent; reconstruct it from the exported library parameters, trial
timing, frame times and shared nuisance matrix. The actual fitted diagnostic
design identities remain in RT provenance. This does not change any fitted
map or score.

The selection-CV maximum remains a selection statistic. The mean-prediction
target assumes transfer of the mean response across runs and does not guarantee
better trial-amplitude estimates. HRF parameters at weak-signal locations have
no significance threshold; 622 candidates were selected and 525 locations were
undefined. The real-data improvement is modest and location-dependent.
