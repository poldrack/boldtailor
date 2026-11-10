# RT effects on HRF recovery: NSD simulations

Status: approved and implemented, 2026-11-10. Synthetic validation and a small
real-input calibration interface check completed; the scientific NSD study
has not been launched. See README.md for executable commands and documented
implementation details (physical boxcar scaling, exact event boundaries,
AR(1) variance projection correction, and calibration limitations).

## Question and scope

How does misspecifying the relation between response time (RT) and neural
activity distort HRF estimation by original REML and AR(1) REML? Generate
single-vertex time series with known HRFs, using actual sub-07 NSD onsets,
binary trial types, RTs, and acquisition times from sessions 10–19. Estimate
noise characteristics from the NSD-preprocessed VTC data. Compare recovered
HRFs with ground truth, rather than treating prediction scores as recovery.

User decisions: use a new `experiments/hrf_simulations/` directory; compare
RT-dependent amplitude and duration; follow the general approach of Mumford
et al. (2024). The numerical defaults below are proposed implementation choices.
Reuse the existing HRF library and REML solvers without changing their defaults.
Keep all new simulation code, tests, reports, and notebook source here.

## Relation to Mumford et al.

Source: Mumford et al., *The response time paradox in functional magnetic
resonance imaging analyses*, Nature Human Behaviour 8, 349–360 (2024),
doi:10.1038/s41562-023-01760-0. Local source:
`../hrf_estimation/references/Mumford_2024_NHB.pdf`.

Follow Figure 2's four distinct fitted task models:

| Model | Condition regressors | Additional common RT regressor |
| --- | --- | --- |
| ConsDurNoRT | Two fixed-duration boxcars | None |
| RTDur | Two RT-duration boxcars | None |
| ConsDurRTMod | Two fixed-duration boxcars | Fixed-duration boxcars weighted by raw RT |
| ConsDurRTDur | Two fixed-duration boxcars | Unit-height RT-duration boxcars |

All columns are convolved with the same candidate HRF. Preserve raw RT in
seconds; do not mean-center it or orthogonalize RT columns against conditions.
RTDur and ConsDurRTDur are different models, not aliases. Existing
`hrf_estimation.design` mode `duration` implements the latter.

The paper generates fixed-duration and RT-duration signals and assesses
condition contrasts under correctly specified and misspecified models. Our
extension explicitly adds amplitude modulation as a generating mechanism and
makes HRF recovery the primary endpoint. Unlike the paper's synthetic RTs,
random trial order, canonical HRF, white errors, and group analyses, we retain
actual NSD behavior/order and use multiple HRFs and empirically calibrated
temporal noise. This is an adaptation, not a reproduction of its error rates.

The paper uses 0.1-second fixed durations in Figure 2. NSD event durations are
3 seconds. Include both **brief (0.1 s)** and **NSD stimulus (observed duration)**
conventions as separately labeled analyses. Do not assume their equivalence.
For pure RT-duration truth, duration is RT regardless of that convention;
simulate it once and fit both fixed-duration conventions to the same data.

## Ground-truth signal

Construct neural boxcars on a high-resolution grid, convolve with known HRF,
then sample at the actual CIFTI frame times. Use an independent synthesis path
and test numerical agreement with fitted designs; a noiseless matched-model
control must detect timing or convolution discrepancies.

Generating mechanisms, retaining every observed valid RT without clipping:

1. **No RT effect:** condition-dependent height, fixed duration.
2. **RT amplitude:** fixed duration, height `beta_condition + gamma * RT`.
3. **RT duration:** unit duration multiplier (height `beta_condition`),
   duration equal to RT. Do not divide boxcars by duration: integrated neural
   activity increases with time on task.
4. **Additive duration:** fixed-duration condition signal plus a common
   RT-duration component of height `gamma`. This gives ConsDurRTDur an exact
   generating-model control in addition to testing it against pure RTDur.

Use equal condition heights `(1, 1)` for the primary study to isolate an RT
confound; unequal heights `(1, 1.25)` form a sensitivity analysis. Use
`gamma = 0.5` for the primary amplitude/additive-duration cases, with `0.25`
and `1.0` as sensitivity levels (amplitude gamma is per second). These are
controlled simulation strengths, not estimates of a causal NSD RT effect.
Keep coefficients fixed across runs in the primary experiment. A later
run-varying-coefficient control can assess the REML random-effects assumption.
The primary gamma values intentionally hold neural heights constant, not the
relative BOLD energy of RT and baseline components. Interpret each duration
convention separately. Add a sensitivity analysis that rescales the RT
component to the same nuisance-adjusted RMS / baseline RMS across conventions
(target 0.5), to distinguish duration mismatch from relative component strength.

Preserve missing/incorrect event categories and include separate junk signals
and fitted junk regressors. Use observed stimulus durations and unit heights
for both junk categories in every mechanism. Missing RT is never imputed to
generate an RT effect. Correct trials with valid positive RT contribute to
the two condition regressors. Keep all time points; no censoring.

HRF truths: canonical, fast, slow, an empirical library HRF, and an off-grid
double-gamma HRF. Specify and save all parameters and sampled curves before
running. Use the existing full 533-candidate library for estimation. Record
the best attainable library approximation under each reported shape metric,
so library discretization is distinguished from estimation error. Normalize
truth HRFs to a common positive peak for signal synthesis and shape reporting;
do not change the existing estimator's candidate scaling.

## Noise and signal calibration

Use VTC vertices from the existing fusiform mask and NSD-preprocessed BOLD and
motion files. Estimate task/nuisance residuals with a fixed flexible reference
design (condition, RT-amplitude, RT-duration and junk terms with fast,
canonical-like and slow HRFs). Report design rank and residual degrees of
freedom. This residual is an empirical noise proxy, not proven pure noise;
task mismatch and finite-sample projection can affect its spectrum.

Use leave-one-session-out calibration: when generating session s, estimate
noise distributions from the other nine sessions. Retain run/vertex profiles
rather than pooling all residual time series into a single signal. Select a
deterministic, saved subset of 64 VTC vertices spanning residual variance and
lag-one correlation for calibration; exclude nonfinite or degenerate series
using documented numerical checks, not model performance.

Noise conditions:

- White Gaussian control at the calibrated marginal variance.
- Stationary AR(1) noise, with variance and regularized lag-one coefficients
  estimated from calibration data. Use projection-bias correction where
  applicable and shrink noisy vertex estimates; record clipping and fitted rho.
- Random-phase surrogates of calibration residuals, preserving the source
  periodogram while breaking event alignment. Preserve real-valued FFT
  symmetry, keep the DC component zero, and handle Nyquist correctly. Match
  donor/target run length and sampling interval; reject incompatible donors
  rather than silently interpolate. This tests non-AR(1) temporal structure.

Assign each replicate one calibration vertex profile, held fixed across its
12 runs. For the primary AR(1) generator use that vertex's regularized pooled
rho and marginal residual variance in all runs, with independent innovations
and stationary initialization per run. Thus the oracle uses one known scalar
rho. For spectral surrogates sample compatible donor runs from that vertex
profile, with independent phases per target run. Run-varying rho/variance is a
separate sensitivity analysis, not a hidden mismatch in the primary AR(1) case.

Validate simulated marginal variance, ACF, and power spectrum against donors.
State that random-phase surrogates assume stationarity and need not preserve
transients, non-Gaussian tails, or the pre-projection noise spectrum. Estimate
AR(1) calibration with bias correction separately from spectrum matching;
do not claim the latter reconstructs noise removed by the reference design.

Estimate a distribution of nuisance-adjusted task RMS / residual RMS in
calibration sessions, with held-out-run task-amplitude predictions used to
reduce in-sample optimism. Define the pilot's signal/noise ratios from its
25th, 50th, and 75th percentiles; save the estimator and numerical values before
simulations. If the calibration cannot yield positive stable values, report
that failure and choose explicitly labeled fixed SNR controls before running.
One common signal multiplier per simulated case sets the target ratio,
computed across all runs after nuisance projection; preserve relative run and
condition effects. Reuse the same noise realization across all fitted models
and paired signal mechanisms. Include noiseless controls separately.

Use actual motion-24, run intercepts, and separately generated cosine terms
with an 80-second HPF cutoff in both generation and fitting. Add a nuisance
signal using reference-fit nuisance coefficients from the calibration donor,
mapped by named columns, only when dimensions match. A zero-nuisance control
checks projection invariance. Do not use fMRIPrep confounds or prefilter the
synthetic signal in addition to including cosines in the model.

## Estimation and evaluation

Cross all four task models with:

- Original iid-noise scalar REML.
- Existing pooled AR(1) scalar REML, estimating rho using only the simulated
  training runs, with a common fixed reference design across task models.

Use one HRF across runs for a simulated vertex. The fixed reference design
for rho includes amplitude and duration terms regardless of fitted task
model, preventing differences in the noise estimator from silently changing
the RT comparison. This is an explicitly documented adaptation of the existing
AR(1) pipeline, whose default reference depends on fitted RT mode. Keep its
projection correction and pooled clipping bounds. Do not pool independent
Monte Carlo replicates to estimate rho. The initial study does not compare
separate-component or shared-effect REML variants.

Fit the complete session for primary HRF recovery. Also fit complementary
within-session six-run halves for stability and transfer checks, using the
same splits as the empirical experiment where available. Calibrated noise
profiles are inputs to the simulator, never supplied to fitted estimators.
An oracle-known-rho diagnostic for AR(1)-generated data separates noise
parameter error from HRF-selection error; it is labeled a diagnostic.

Primary outcomes: unit-peak HRF curve RMSE, curve correlation, signed peak-time
bias and absolute peak-time error, and failure/zero-variance frequency. Report
all replicates, not just converged fits; summarize recovery conditional on
success alongside unconditional failure rates. Exact candidate recovery is
secondary and only meaningful for in-library truths. Report Monte Carlo
uncertainty and paired differences between estimators on identical data.

Secondary outcomes: rho estimation error when defined, fit diagnostics,
split-half shape consistency, and held-out signal reconstruction. For
reconstruction, compare fitted predicted task signal to the known noiseless
task signal as well as observed-data scores. Clearly distinguish refitting
amplitudes on held-out runs (HRF-only transfer) from fully held-out prediction.
The primary claim concerns HRF recovery, not group-level type-I error or
identification of a unique biological RT mechanism.

## Staging and outputs

1. Inventory all sessions 10–19 and persist timing/RT summaries and hashes.
   Calibrate noise, validate spectra/ACF, and freeze simulation parameters.
2. Tests first, committed while failing, then implementation. Include analytic
   boxcar checks, real-RT preservation, no cross-run convolution, noise
   stationarity/variance checks, matched-model noiseless recovery, exclusion
   of target-session calibration, seed reproducibility, and solver parity.
3. Pilot session 11, canonical and off-grid truths, four generating mechanisms,
   both duration conventions, median SNR, three noise families, 20 paired
   replicates. Run eight fitted models per fixed-duration convention (four
   task models times two noise assumptions); reuse RTDur fits when identical
   across conventions and reuse pure-duration generated data. Profile runtime
   before selecting workers; cache candidate designs within valid timing/model
   groups. Save checkpoints atomically and resume only matching manifests.
4. Expand to sessions 10–19, five HRFs, three SNR levels, 100 replicates for
   the primary settings after pilot checks. Run strength/condition-contrast
   sensitivities as separately labeled subsets, not an uncontrolled full cross.
   Report expected runtime and storage from the pilot before expansion.
   The pilot is the initial execution scope; the large expansion is not
   automatically submitted. Review its measured cost and scientific results
   first. Required pilot checks: reproducible paired seeds, preserved source
   events, no calibration leakage, numerical agreement of independent
   synthesis and matched design, noise variance/ACF within Monte Carlo
   uncertainty of generator expectations, and explicit accounting of every
   failed fit. Poor HRF recovery is a result to investigate, not a reason to
   discard cases or tune truth/noise until a preferred method succeeds.
5. Create a Python percent-format notebook and local rendered notebook with
   model diagrams, actual RT summaries, noise validation, true-versus-recovered
   curves, paired error plots, and failure rates. Track Python notebook and
   compact reports; keep rendered notebooks and bulky simulation outputs local.

Manifest must record source files/hashes, calibration session and vertex IDs,
candidate library fingerprint, truth curves, mechanisms and duration conventions,
coefficients, SNR definition/values, noise donor/parameters, split/run IDs,
seed hierarchy, solver settings, software revision, and completion hashes.
Keep residual calibration data local. A report should be viewable without the
NSD drive, using committed aggregate summaries and selected HRF curves.

## Interpretation limits

Amplitude and duration can yield very similar BOLD signals; apparent HRF
recovery does not establish which neural mechanism generated empirical data.
Long RTs and the 3-second stimulus convention make this especially important.
REML compares covariance models and its shared variance component is sensitive
to regressor scaling: keeping raw RT and native solver scaling reproduces the
analysis of interest, but differences need not reflect omitted-signal bias
alone. Different numbers of random effects are part of the model comparison.
Noise calibration is from one participant and VTC, not all brain regions.
