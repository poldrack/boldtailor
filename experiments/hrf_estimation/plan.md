# Empirical HRF estimation experiments

Detailed planning: [Implementation and validation plan](implementation_plan.md) (2026-11-07). The Q&A below records confirmed decisions; the detailed plan distinguishes these from proposed implementation defaults and prerequisites.

I am interested in running some experiments to find the best way to estimate empirical hemodynamic responses at each voxel in fMRI data. The model needs to take into account the known relationship between response times and hemodynamic responses, ensuring that response times differences are factored into the HRF estimation procedure (see hrf_estimation/references/Mumford_2024_NHB.pdf).  It should also take into account trial type differences, since stimulus repetition can have substantial effects on HRF responses.  

One note: The NSD dataset contains 7T fMRI data which have substantially faster HRFs than the standard canonical HRF commonly used for 3T data.

## Data:

We will use the Natural Scenes Dataset. The data are in /Volumes/extdata1/NSD - for these experiments we will focus on the NSD preprocessed data for sub-07, which are in /Volumes/extdata1/NSD/BIDS/derivatives/nsdpreproc/sub-07.  We will use sessions 11-20, using the CIFTI data (e.g. sub-07_ses-nsd11_task-nsdcore_run-09_space-fsLR_den-32k_desc-layerB2_bold.dtseries.nii) and the events files in the standard BIDS location (e.g. BIDS/sub-07/ses-nsd10/func/sub-07_ses-nsd10_task-nsdcore_run-09_events.tsv).

Use the motion confounds from the NSD-preprocessed derivatives, matched to each run (e.g. BIDS/derivatives/nsdpreproc/sub-07/ses-nsd11/func/sub-07_ses-nsd11_task-nsdcore_run-12_desc-confounds_timeseries.tsv). Do not use the fMRIPrep confounds: the NSD-preprocessed BOLD data were temporally resampled. Construct the agreed 24-parameter motion model from the NSD-preprocessed motion parameters, verifying alignment with the corresponding CIFTI time points. Generate cosine basis functions separately within the model on each run's CIFTI sampling grid for high-pass filtering.

## Output:

Experiments should be performed within experiments/hrf_estimation.  Use a BIDS-like naming scheme for all outputs when possible.

## Methods:

There are several methods that I want to compare:

1.  Crossvalidation with HRF library: the boldtailor package currently implements selection from a library of HRFs using crossvalidation.
2.  REML estimation with HRF library: there is a prototype of this in ../boldtailor_testing (see src/ for source and analysis/hrf_offset0 for some example usage) - hasn't yet been implemented within boldtailor
3.  Regularized estimation of HRF parameters: This should directly estimate the optimal parameters for the SPM hrf function using optimization.  see references/pedregosa2015.pdf for an example - but think broadly about the optimal way to accomplish this, and test multiple optimization methods if it makes sense.

All methods should include the following features in the model for estimation:
- trial_type factor
- response_time modulation (common across all trials)
- motion confounds from the NSD-preprocessed data, using a 24-parameter motion model
- high-pass filtering with an 80-second cutoff (0.0125 Hz), using cosine basis functions generated within the model on each run's CIFTI sampling grid


## Validation targets

there are two primary goals here: 
- in-sample r-squared: using an encoding model with all of the features described above, quantify the variance accounted for within each voxel
- out-of-sample r-squared: using shuffle split, fit the model to half of the runs and estimate r-squared using the selected HRF across the remaining runs

there are also secondard goals:
- HRF stability (within-session): how stable is the estimate across random subsets of runs within a session?
- HRF stability (between-session): how stable is the estimate across sessions?

## Planning clarification Q&A (2026-11-07)

The following answers clarify the experimental plan above.

1. **Question:** On held-out runs, should both the HRF and task coefficients be frozen, or only the HRF?
   **Answer:** Freeze only the HRF; refit task coefficients on held-out runs. The held-out evaluation therefore measures HRF transferability.

2. **Question:** Should the 50/50 run splits occur within each session or pool runs across sessions 11–20?
   **Answer:** Use within-session splits.

3. **Question:** Should trial types and the common response-time effect share one HRF per location, with separate amplitudes?
   **Answer:** Yes, use one shared HRF per location.

4. **Question:** Should trial type use the binary `trial_type` column alone, or distinguish additional repetition categories?
   **Answer:** Use binary `trial_type` only.

5. **Question:** Should R² measure task-related variance beyond nuisance terms or the complete model including nuisance terms?
   **Answer:** Report both.

6. **Question:** Which nuisance regressors and missing/incorrect-response policy should be used?
   **Answer:** Include cosine drifts and motion parameters using a 24-parameter motion model, with no censoring. Code missing responses and incorrect responses as separate junk regressors: one for missing and another for incorrect.

7. **Question:** What spatial and computational scope should be used for initial testing?
   **Answer:** Testing can begin with a limited subset focused on ventral temporal cortex, since it should respond well to these tasks. The exact region definition, full spatial scope, and compute/runtime budget were not specified in this answer.

8. **Question:** Should REML and continuous HRF optimization become reusable package functionality now or remain experimental?
   **Answer:** Leave REML and continuous HRF estimation as experimental implementations under `experiments/hrf_estimation` for now.

## Confound-source amendment (2026-11-07)

Following inspection of the different temporal sampling grids, the user specified that motion confounds must come from the NSD-preprocessed data rather than fMRIPrep. Cosine basis functions for high-pass filtering will be generated separately within the model. This supersedes the original fMRIPrep confound-source choice and the proposal in `implementation_plan.md` to interpolate fMRIPrep confounds onto the CIFTI grid. The 24-parameter motion model, no-censoring policy, and separate missing/incorrect-response junk regressors remain in effect. The user subsequently specified an 80-second high-pass cutoff (0.0125 Hz).

