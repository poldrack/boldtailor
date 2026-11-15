# Scientific-validity re-review of REML and AR(1)

Date: 2026-11-15  
Reviewed commit: `ac2d870`  
Verdict: **NOT READY to close all scientific-validity findings.**

The earlier counterexamples are fixed, and the implementation and tests are
substantially stronger. This pass nevertheless establishes two further
numerical/statistical defects: an inaccurate orthogonal residual sum of squares
can distort REML estimates and likelihoods, and an uninformative AR(1) moment
curve can produce an apparently valid extreme autocorrelation estimate.
Experiment adapters also discard part of the new scientific uncertainty
information. These are not packaging, performance, or other operational issues.

## Scope and claim surface

Reviewed the original review, both responses, the follow-up assessment, the
second red-team report, changes since `87affe6`, and the current REML/AR(1)
implementation and relevant tests. Inspected the immediate HRF experiment
adapters to see whether the new uncertainty flags reach scientific outputs.
This is a full re-review of that numerical subsystem, not a fresh audit of
every model or experiment elsewhere in the repository.

The statistical model remains a shared scalar noise variance and a shared
scalar random-effect variance, independent group-specific effects, and supplied
AR(1) rho. The reduction to a whitened, fixed-effect-restricted likelihood is
sound in exact arithmetic. Candidate comparisons require the same fixed design
and rho. A positive variance estimate or a selected candidate is not, by itself,
a calibrated significance test.

| Readout | False-positive route | False-null / attenuation route | Evidence or protection |
| --- | --- | --- | --- |
| Variance components and status | Arbitrary split under confounding; numerical remainder interpreted as noise | Real residual removed by projection policy or variance boundary comparison | `_reml_search._confounded`; prior probes now pass; remaining B1 below |
| Restricted criterion and candidate winner | Inaccurate criterion or unresolved optimization favors a candidate | Missed basin or truncated competing candidate hides a better model | Independent covariance tests; all-crossings search; B1 and M1 below |
| Corrected and pooled rho | Flat/ambiguous correction curve maps to an arbitrary extreme rho | Clipping, projection, or invalid reference suppresses real autocorrelation | Whitening property test; moment tests; B2 below |
| HRF estimate and reported uncertainty | Provisional winner reported without its qualification | Undefined/unresolved results treated as substantive absence | Status propagation partly retained; M1 below |
| Null calibration and parameter recovery | Plug-in rho/reference misspecification distorts calibration | Same choices reduce recovery despite a real random effect | Existing oracle-rho Monte Carlo checks; proposed end-to-end checks below |

## Blockers for unrestricted scientific use

### B1. Orthogonal residual energy is obtained by unstable subtraction

**Classification: Defect.** The reported criterion disagrees with the model's
independently evaluated covariance, and changes under an iid observation
permutation. These violate the stated likelihood, not a preference for a
different estimator. **Priority: high. Direction: either**—noise estimates and
candidate likelihood comparisons can be distorted.

Location: [`reml_terms`](../src/boldtailor/_reml_terms.py), particularly line 128:

```python
residual_ss = np.maximum(energy - coordinates2.sum(axis=0), 0.0)
```

The subtraction loses the small orthogonal remainder when most energy lies in
the random-design span. Clipping negative differences to zero does not repair
positive subtraction error or recover a small genuine remainder. Because this
remainder is not shrunk by lambda, it can dominate the likelihood at large
ratios even when its error is tiny relative to the original signal energy.

**Executed saturated example.** With no fixed effects, iid noise,
`s = logspace(-4, 4, 10)`, `Z = diag(s)`, and `y = s[:, None]`, random rank equals
the entire restricted dimension. The exact orthogonal remainder is zero, but
the code produces `1.4901161193847656e-8`.

| Quantity | Observed |
| --- | --- |
| Returned status and flags | `ok`, `at_upper=False`, `unrefined=False` |
| Returned ratio | `3.539236568e7` |
| Reported criterion | `1.128158267` |
| Independent diagonal-covariance criterion at that same ratio | `0.571657282` |
| Independent criterion at feasible ratio `1e16` | approximately zero |
| Criterion after reversing observation rows of both y and Z | approximately zero |

There is no dropped direction and no fixed-design uncertainty in this example.
The diagonal oracle uses elementwise positive variances, so it does not rely
on an ill-conditioned dense matrix inversion. A better feasible point suffices
to disprove the returned optimum; no global optimization claim is needed.

**Executed positive-noise example.** Append an eleventh zero row to Z and a
signal value `1e-5` to y. The true orthogonal energy is now `1e-10`, but the
stored remainder is still `1.490116119e-8`. The fit reports ratio `8.3985e7` and
criterion `-17.75829525`; independently evaluating the feasible ratio `1e10`
gives `-23.02580619`. A separate scalar search of this diagonal objective puts
its minimum near `9.9909e9`. Thus this is not confined to a zero-noise boundary:
it can move a finite variance-ratio estimate by over two orders of magnitude.

**Required correction.** Compute orthogonal residual energy stably, preferably
from orthogonal-complement coordinates or an appropriately stable residual
projection. Enforce the structural zero remainder when the retained random
directions fill the restricted space. Do not solve only the full-rank case:
the eleven-observation example requires preserving real small residual energy.
Numerical uncertainty that cannot be resolved should not yield an unqualified
ordinary estimate.

**Verification.** Independent diagonal/dense covariance checks with random rank
at, just below, and far below restricted dof; controlled residual-to-random-span
energy ratios; iid row permutations and general orthogonal row transforms;
explicit criterion comparisons at the returned ratio and at known feasible
alternatives. Compare estimates only when their optima are identifiable and
interior; use likelihood/covariance comparisons at boundaries.

### B2. An uninformative AR(1) moment curve yields a fabricated extreme estimate

**Classification: Defect.** A constant statistic-to-parameter curve cannot be
inverted to identify rho. Choosing its first grid point does not estimate rho.
**Priority: medium. Direction: either**, through false autocorrelation and the
resulting whitening or variance estimates.

Location: [`invert_lag_ratio`](../src/boldtailor/_ar1_moments.py), lines 83–95,
and its use by `ar1.estimate_rho` and `pool_rho`.

**Executed public-API example.** For `I = eye(10)`, take reference design
`I[:, 1:]` and signal `I[:, [0]]`. The residual is nonzero, but its lag
numerator is identically zero. The entire expected-lag-ratio curve is exactly
zero for every rho on the correction grid. Current results are:

```text
raw ratio = 0
corrected rho = -0.95
pooled rho = -0.8
```

The nonmonotone branch's nearest-point `argmin` selects the first of 2,001
equally good grid points, and pooling treats that choice as a finite estimate.
This can also occur when several residual coordinates survive but none are
adjacent; it is not inherently a zero-residual problem. For the saved example,
the sole remaining observation has marginal variance independent of rho.

The strongest rationale for nearest-grid inversion is that a nonmonotone
curve still needs a practical point estimate. That rationale does not apply
to a flat curve with no information at all. Multiple separated inverse
solutions also need a stated ambiguity policy rather than an arbitrary index
tie-break being mistaken for identification.

**Required correction.** Detect nonfinite, flat, and non-identifying correction
curves before returning a corrected rho. Return undefined/NaN or an explicit
ambiguity result and keep such values out of ordinary pooling. The current
documented no-finite-values pooled fallback of zero can remain a declared
fallback; it must not be confused with an estimated rho.

**Verification.** Generate reference subspaces with one residual dimension,
nonadjacent residual coordinates, near-flat moment curves, and well-identified
curves. Assert that genuine invertible cases recover grid rho while
uninformative cases do not produce finite estimates. Keep amplitude degeneracy
and parameter non-identifiability as separate cases.

## Majors and remaining scientific qualifications

### M1. Experiment outputs drop the new selection qualifications

**Classification: Defect in uncertainty reporting.** A selection explicitly
declared provisional by the numerical API loses that qualification in the
experiment result. **Priority: medium. Direction: either**—the apparent
confidence in a selected HRF can exceed the resolved comparison.

Locations: `experiments/hrf_estimation/reml_selection.py::_diagnostics` and
`experiments/hrf_simulations/fitting.py::_fit_diagnostics` / `fit`.

Both adapters export the winning fit's `at_upper`, but neither exports
`selection.any_at_upper` or `fit.unrefined`. The latter simulation helper only
receives the winning fit, so it cannot preserve information about a truncated
losing candidate. I executed both diagnostic helpers on an actual truncated
selection and confirmed the new fields are omitted. Inspection establishes the
loss for losing-candidate truncation too; I did not demonstrate a resulting
wrong HRF winner on real data.

Carry the qualifications into saved scientific results and interpret affected
selections as provisional. An end-to-end check should assert that a provisional
comparison remains distinguishable from a resolved one in the reported result.
This tests scientific interpretation, not merely object serialization.

At the core API level, `any_at_upper` is a substantial improvement. A losing
interior `unrefined` candidate still has no aggregate uncertainty flag. This
known limitation deserves the same treatment if that fallback occurs; no new
default-grid example of it was found in this review.

### M2. Plug-in rho and approximation robustness remain incompletely validated

**Classification: Undetermined.** Existing evidence establishes useful cases,
but does not settle the end-to-end calibration under estimated rho or model
misspecification. This is a testing gap, not an established error.
**Direction: either.**

The AR(1) correction is explicitly a ratio-of-traces approximation to an
expectation of a ratio. Tests verify the trace algebra and successful recovery
for particular reference designs. REML recovery under AR(1) uses the true rho;
the finite-sample null test uses iid noise. These do not establish null
calibration after rho is estimated from the same data and a possibly incomplete
task/reference basis. See the proposed Monte Carlo studies below.

**Contested choices, not defects:** pooled/clipped rho, a shared scalar noise
variance, and iid random coefficients are defensible modeling assumptions.
Heterogeneous run variances, group-dependent rho, correlated random effects,
or signals outside the reference span violate some of them. Sensitivity to
those alternatives should define the intended scientific operating range,
not be labeled automatically as an implementation failure.

## Disposition of earlier findings

| Earlier item | Current assessment |
| --- | --- |
| R1, exact variance-component confounding | Fixed for the retained-spectrum contract; NaN components and exclusion from selection are appropriate. Near-confounding is explicitly weak information, not a solved estimation problem. |
| R2, missed bracketed basin | Saved counterexample passes; all-crossings refinement and spread-dependent grid are improvements. The documented lack of a certified global search remains. B1 is a different failure upstream of the search. |
| R3 / F1, fixed shifts and column units | All saved probes pass. Column normalization and distinct `unresolved` status address the demonstrated regression. The new numerical property tests materially improve coverage. |
| F2, unqualified grid fallback | `unrefined` makes the individual-fit qualification explicit. Selection-wide and adapter propagation remain incomplete as described in M1. |
| R4/R5, scaling and memory | Prior structural fixes remain present. No further operational testing is recommended in this scientific review. |
| R6, common numerical policy | Partial migration is accurately documented. Different policies elsewhere need cross-method scientific checks where the same fixed span is intended, not consolidation merely for its own sake. |
| Second review, zero slope and extreme-scale helpers | Existing regression tests pass. Local norm/floor fixes are not a proof of end-to-end variance estimation at every floating-point scale. |

## Property-based testing with scientific value

`tests/test_numerics_properties.py` already covers the shared numerical helpers
extensively. `tests/test_reml_properties.py` already tests public-API
invariances, and `test_reml_search.py` has a generated best-feasible-grid check.
`test_ar1.py` uses Hypothesis for whitening covariance. The recommendations
below distinguish new direct coverage from extensions to existing coverage.

| Priority / functions | Current coverage | Valuable generated property and scientific purpose |
| --- | --- | --- |
| **1 — `reml_terms`, `_random_directions`** | No dedicated generated independent-oracle tests; exercised indirectly by public REML properties. Fixed examples dominate the direct tests. | Generate known restricted spectra, random ranks from 0 through full restricted rank, and independently specified orthogonal residual energy. Compare remainder and quadratic forms with known coordinates, including high SNR. Exposes B1 and rank-induced loss of signal. |
| **1 — `moment_polynomials`, `basis_expected_lag_ratio`, `invert_lag_ratio`, `estimate_rho`** | Direct tests are examples/parametrizations; the AR(1) Hypothesis test covers `whiten`, not these scientific properties. | Generate subspaces and rho, compare polynomial moments against independently formed dense traces, and invert only when the curve identifies rho. Flat/ambiguous curves must not yield apparently identified estimates. Test joint signal/reference rescaling and fixed-span shifts through the public estimator. |
| **1 — `pooled_quadratic`, `pooled_criterion`, `pooled_slope`, `profiled_variance`** | Direct criterion tests use one simulated design per rho and a short fixed ratio list; indirect generated search coverage exists. | Generate positive spectra, ranks, coordinate energies and residual energies; compare against an independent diagonal or restricted dense covariance. Check Q is nonincreasing in nonnegative lambda, positive where appropriate, and the log-lambda derivative matches an independently differentiated objective. Cover repeated and highly separated eigenvalues and multiple groups. |
| **2 — `_confounded`** | Strong hand-picked cases, but no dedicated generated confounding classification property. | Generate B = cI in restricted coordinates, including whitening, fixed effects, group scales, and implicit zero directions. Common c across all groups must be unidentifiable; differing c or a retained zero direction must not be declared exactly confounded. Keep weak identification separate from exact classification. |
| **2 — `fit_reml` / `minimize_ratio`** | Property tests already exist, mostly low-rank, well-conditioned iid simulations. | Extend to rank near saturation, noise-to-effect ratios near boundaries, and negative/nonzero rho. Check iid row orthogonal invariance, and AR(1) time reversal, against independent criteria. Check a returned ordinary solution is no worse than independently chosen feasible ratios. Boundary plateaus require likelihood/covariance tolerances, not equality of arbitrary huge ratios. |
| **2 — `select_random_design` / `_compete`** | Candidate-order Hypothesis coverage and a fixed brute-force comparison already exist. | Generate mixed identifiable, null, unidentifiable and provisional candidates. Compare eligible likelihood ranking with independent per-candidate calculations; preserve qualification when a losing candidate is unresolved by the search. Check global random-design scaling equivalence and covariance-preserving factor changes. |
| **3 — `project_fixed`, `reproject_signals`, `_above`** | Extensive lower-level numerical properties and fixed public status tests. | Generate mixed live, unresolved and zero features across groups and rho subsets. Require independent fitting and joint fitting to agree scientifically on estimates, statuses and dof. Include rank-borderline designs, but assess uncertainty rather than demanding exact invariance across a numerical rank transition. |

Specific limitations in current generators:

- Public REML property parameters restrict fixed/random counts to 1–4 and
  scans to 12–40, and simulations use rho zero. They do not reach the saturated
  random designs or general AR(1) cases involved above.
- `_assert_same_fit` omits the recently added `unrefined` flag. Scientific
  equivalence should include whether the returned estimate is qualified.
- The `repeated` spectrum generator in `test_reml_adversarial.py` uses
  `repeat(two_values, k)[:k]`, yielding k copies of the first value. It covers
  isotropy, but not two distinct repeated spectral clusters.
- `test_energy_decomposes` reconstructs energy from the same subtraction used
  by the implementation. It cannot establish that the tiny remainder is
  independently correct. A relative tolerance against total energy can hide
  a large relative error in the noise-relevant remainder.

Use valid symmetries: arbitrary row permutations/orthogonal row transforms
preserve iid REML, but not a fixed AR(1) temporal covariance. Time reversal
does preserve that AR(1) covariance. Global scaling or right-orthogonal rotation
of Z has a known transformation law; independently scaling its columns changes
the iid random-effects model and should not be required to leave a fit unchanged.

## Additional scientific tests beyond algebraic properties

1. **Estimated-rho end-to-end calibration.** Simulate null and known-positive
   random effects; estimate rho using the actual reference path, then fit/select.
   Compare against known-rho fits over scan counts, fixed ranks, positive and
   negative rho, and reference bases that do/do not contain the task signal.
   Assess noise/effect recovery, null likelihood-ratio distributions, clipping,
   and undefined-estimate rates with predeclared Monte Carlo tolerances.
2. **Near-confounding and boundary behavior.** Vary spectral distance from cI,
   residual dof, and true noise variance toward zero. Track uncertainty flags,
   estimated covariance and criterion error as well as component estimates.
   A wildly varying split in a weakly identified model is not alone a code bug;
   false precision or wrong likelihood is the concern.
3. **Candidate-selection calibration.** Under no random effect, vary library
   size and candidate correlation and report how often any candidate wins.
   Under known effects, assess recovery and ties. This bounds the meaning of
   a winner or a gap; it must not assume maximum likelihood selection itself
   controls a false-positive rate.
4. **Model-assumption sensitivity.** Examine unequal run variances/rho,
   correlated random effects and reference misspecification if those occur in
   the intended experiments. Predefine acceptable operating conditions and
   report failures as robustness limits unless the model promises otherwise.
5. **Scientific adapters.** Verify provisional/unresolved results remain
   qualified in saved HRF results and downstream summaries. Check that the
   reference used to estimate rho and the fixed span used to estimate residual
   variance agree where the experiment assumes they are the same; column-unit
   changes should not silently change that assumption across fitting paths.

These recommendations concern numerical correctness, identification,
calibration and interpretation. No memory, speed, cache, packaging, input-error
message or other operational test additions are proposed.

## Checks: executed versus inspected

- **Executed:** targeted REML, AR(1), numerical and numerical-property tests,
  plus the five previously saved reviewer probes: **722 passed, 9 skipped,
  2 xfailed**. The skipped slow statistical cases and existing expected failures
  were not claimed as passes.
- **Executed:** four immediate experiment test files through
  `uv run --no-sync python -m pytest`: **28 passed**. An initial invocation via
  the pytest executable could not import `experiments`; the module invocation
  above completed successfully.
- **Executed:** five new probes in
  [`test_scientific_rereview.py`](reml-review-2026-11-15-reproducers/test_scientific_rereview.py):
  **5 failed for their intended scientific assertions**, four addressing B1
  and one addressing B2. The diagonal oracle does not use production sufficient
  statistics. These probes are outside the default test suite.
- **Executed:** independent positive-noise diagonal objective search and
  explicit feasible-point comparisons described under B1; direct public AR(1)
  constant-curve example under B2; adapter diagnostic-field inspection under M1.
- **Inspected:** likelihood derivation, rank and status policies, search
  qualification, prior fix history, test generators/oracles, and adapter
  propagation. No implementation changes or test weakening were made.

## Limits of this assessment

This is not a certificate for all scientific uses. The demonstrated defects
use accepted adversarial inputs; their prevalence in ordinary fMRI designs was
not estimated. No real-data result was reanalyzed, and no wrong real-data HRF
winner is asserted. The search remains practical rather than globally certified.
The slow Monte Carlo suite, arbitrary noise misspecification, and the broader
package's separate fitting policies were not exhaustively audited. A claim
that a selected HRF is biologically correct or statistically significant needs
additional construct-level and inferential validation beyond this subsystem.
