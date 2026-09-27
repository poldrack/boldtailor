# NSD workflow notebook validation

The [full workflow notebook](../../examples/NSD/nsd_workflow.ipynb) was
executed on 2026-09-27 using all 12 runs of `sub-07/ses-nsd10`, all 750
trials, and the first 128 grayordinates. It used the full 649-candidate HRF
library, two workers, blocks of 64 grayordinates, OLS conventional GLMs,
and canonical/optimized single-trial models with OLS and fixed ridge 0.1.

Of the 128 requested grayordinates, 123 had nonconstant signals. Outputs
preserved the complete 91,282-grayordinate CIFTI axis, with NaN at constant
and unprocessed locations. The notebook published 125 files to a temporary
validation directory. It did not modify existing NSD derivatives.

Retained scan counts after leading nonsteady-volume removal were:
`187, 186, 187, 187, 187, 187, 185, 187, 187, 186, 187, 187`.
Original acquisition times and event onsets were retained.

An independent NumPy least-squares audit reconstructed both conventional
GLMs from the exported per-run/per-HRF designs and original signals:

| Maximum absolute error after float32 export | Canonical GLM | Optimized GLM |
| --- | ---: | ---: |
| Task/RT/trial-type effects, native units | 5.62e-7 | 8.91e-7 |
| Pooled full R² | 2.97e-8 | 2.90e-8 |

The audit also verified full-minus-confounds R², optimized-minus-canonical
R², exact library lookup for all/odd/even parameter maps, and all 750 trial
identities in each of four beta-series models. Saved figures were inspected
for readable labels and clear interpretation of the descriptive RT checks.

Nine automated tests cover event coding, invalid inputs, scan trimming,
source identity, numerical agreement for serial and parallel GLMs, full
notebook execution against a four-run CIFTI fixture, artifact protection,
and recomputation after changing the ridge penalty. The fixture uses a
small HRF library to keep the tests fast; the real-data check uses all 649.

This validates the workflow on a spatial subset, not full-brain notebook
runtime or peak memory. Earlier whole-brain command-line measurements are
recorded separately in the [session validation](nsd-session.md).

## Full-HRF curve comparisons

The added curve-comparison cell was also run against the existing whole-brain
command-line odd/even HRF maps. All 90,757 grayordinates with both HRFs defined
were compared on the full library time grid. Median Pearson correlations were
0.81290 for odd/even, 0.70968 for odd/canonical, and 0.68936 for even/canonical.
These are separate medians, not paired differences or significance estimates.
The source maps use the command-line scan handling described in the session
validation; these numbers are not a new run of the notebook's trimmed models.

An independent `numpy.corrcoef` check of 256 sampled grayordinates agreed
within 1.45e-15. The notebook's new histogram and baseline comparison plots
were inspected, and fixture execution verified the exported three-map CIFTI
against direct curve correlations. Missing selections preserve pairwise NaN
values. Beta-series progress was checked to remain at two lines per model
as the number of blocks increases.
