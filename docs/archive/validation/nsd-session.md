# NSD session validation

> **Dated validation record:** Results and test counts apply to the recorded
> revision/settings, not necessarily this checkout. See the
> [documentation index](../README.md) for current API and methods guidance.

> **Note (2026-10-02):** Recorded beta magnitudes below predate per-event
> unit-peak normalization. They were fitted with sum-to-one kernels and unit
> event amplitudes; current betas are the peak response to each presentation,
> so their magnitudes differ by an event- and kernel-dependent factor. For a
> 3 s canonical SPM event at TR 1.6 s and oversampling 50, the event response
> peaks at 87.3 against a kernel sum of 148.2, so current betas are about 0.59
> times the recorded ones (the kernel sum-to-peak ratio alone, about 148, is
> not the beta factor). HRF selections, R², and fractions are scale-invariant
> and essentially unaffected (per-event realized-count scales vary by ±1
> sample).

These measurements describe the September 2026 analyses of `sub-07/ses-nsd10`.
For running an analysis, see the [NSD guide](../../examples/NSD/README.md).

## Verified sub-07/ses-nsd10 run

The 2026-09-26 run fitted all 12 runs, 750 trials, and 91,282 grayordinates
with OLS and fixed ridge alpha 0.1. It published 66 new files under
`/Volumes/extdata1/NSD/BIDS/derivatives/boldtailor/sub-07/ses-nsd10/func/`.
Checksums confirmed that all 21 pre-existing files were preserved.

| Pooled diagnostic (median across defined grayordinates) | OLS | Ridge 0.1 |
| --- | ---: | ---: |
| Full R² | 0.818498 | 0.799512 |
| Confounds-only R² | 0.569213 | 0.569213 |
| ΔR² | 0.222574 | 0.205088 |

There are 525 grayordinates with undefined pooled R². Single-trial models
have many more task coefficients than the conventional model, so increased
in-sample R² alone is not evidence of better generalization.

The five vertices selected by odd-run OLS had even-run correlations
`0.163, 0.164, 0.172, -0.154, 0.173`; fixed ridge at the same vertices gave
`0.225, 0.217, 0.222, -0.139, 0.187`. All retained their odd-run sign.
The spatial correlation between odd/even cortical RT maps was 0.657 for OLS
and 0.750 for ridge. These are modest, directionally consistent RT associations,
not significance estimates or proof that ridge is generally superior. The
scatterplots show a sparse long-RT tail; no outliers were removed or used to
choose a penalty.

An independent NumPy audit reconstructed 30 sampled grayordinates (including
the five selected vertices) across every run using least squares and augmented
least squares. Maximum absolute beta differences were below `7.7e-6` native
units, and pooled R² errors below `3e-8`, consistent with float32 storage.
The audit also checked exact imaging axes, trial identities, RT correlations,
valid-trial counts, and odd-only selection. The measured run took 100.57 seconds
with 2.57 GB maximum resident memory (decimal GB, macOS `/usr/bin/time -l`).


## Verified expanded sub-07/ses-nsd10 run

The 2026-09-26 expanded analysis fitted all 12 runs, 750 trials, and 91,282
grayordinates. It published 72 new files in the same derivatives directory;
all 87 earlier files were checksum-preserved. Runtime was 720.54 seconds
(12.01 minutes), with 6.39 GB peak resident memory. The run used 4096-feature
blocks and candidate batches of 32, without changing the library or ridge
penalty after evaluation.

| Median nuisance-adjusted mean-prediction R² | Selected HRF | Canonical SPM | Paired ΔR² |
| --- | ---: | ---: | ---: |
| All-run selection CV (used to choose HRFs) | 0.005495 | 0.000952 | 0.003686 |
| Independent even runs (odd-trained HRF and amplitude) | 0.002465 | 0.001078 | 0.000924 |

Each column is a separate median; the median paired difference need not equal
the difference of medians. Independent prediction improved at 57.93% of defined
grayordinates. There were 525 undefined HRFs and 622 distinct selected candidates.
The improvement is small and varies across locations; the larger selection-CV
gain is not an independent estimate of predictive benefit.

| Median pooled single-trial diagnostic | Optimized OLS | Optimized ridge 0.1 |
| --- | ---: | ---: |
| Full R² | 0.812488 | 0.799636 |
| Confounds-only R² | 0.569213 | 0.569213 |
| ΔR² | 0.219646 | 0.207429 |

The mean-prediction selection target does not maximize single-trial in-sample
R²; the optimized OLS median is slightly lower than the canonical OLS median.
At the same five canonical odd-RT-selected vertices, independent even-run
correlations were `0.229, 0.197, 0.197, -0.102, 0.150` for optimized OLS and
`0.251, 0.247, 0.213, -0.107, 0.135` for optimized ridge. Three associations
strengthened and two weakened relative to their canonical counterparts;
all five retained their odd-run sign. These are descriptive checks.

An independent audit reconstructed 32 grayordinates across every run using
direct convolution, explicit held-out predictions, and NumPy ordinary/augmented
least squares. Maximum beta error was `3.80e-6` native units; R² errors were
below `3e-8`, consistent with float32 storage. It also verified the saved
designs, exact CIFTI axes, all 750 trial identities, RT vertex selection and
even-run correlations. Library and diagnostic plots were visually checked.
See the [validation record](../superpowers/validation/2026-09-26-hrf-selection.md)
for audit details, implementation decisions, and the remaining archive limitation.

The subsequently added `hrfdeltarsquared` maps were generated from the saved
full-model R² images without refitting. Matching axes, all 750 trial rows,
confounds, source runs, and estimator settings were verified first. Every
grayordinate was checked against direct subtraction, with 525 undefined values
per map. Median optimized-minus-canonical R² was −0.002160 for OLS (34.36%
positive) and +0.001326 for ridge (59.61% positive). The two maps and their JSON
sidecars preserve all 160 files present before this addition.

## Parallel fitting benchmark

On the 16-core, 128-GiB development machine, a real-data benchmark using all
12 runs, all 649 HRFs, and 16,384 grayordinates measured:

| Workers | Fitting time | Speedup | Sampled peak process-tree RSS |
| --- | ---: | ---: | ---: |
| 1 | 186.19 s | 1.00× | 5.43 GB |
| 2 | 104.66 s | 1.78× | 9.19 GB |
| 4 | 74.22 s | 2.51× | 12.77 GB |

Four workers are a reasonable starting point on this machine. These timings
include process startup and cold design caches, OLS/ridge fitting, canonical
comparisons, and block-level odd/even checks. They exclude final artifact
assembly/publication and cover four 4096-feature blocks, not the entire
session. RSS is the sampled sum across the parent and children (decimal GB);
shared pages may be counted more than once. Every benchmark map and trial beta
matched serial execution exactly. See the
[parallel validation record](../superpowers/validation/2026-09-26-nsd-parallel.md).

## Odd/even HRF maps

The later split-HRF export added even-run HRF indices and matched odd/even
parameter maps. It verified independent selections, matching spatial axes,
and preservation of earlier derivatives. See the
[split-HRF validation record](../superpowers/validation/2026-09-27-split-hrf.md)
for the generation settings and checks.
