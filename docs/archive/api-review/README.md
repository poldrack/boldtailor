# Complete package inventory

[Review guide](../code-review.md) · [User-facing API](../api.md)

Snapshot of the source tree on 2026-11-15. Every shipped Python module is listed, including empty initializers. Each module page contains its locally defined functions, classes, methods, properties, nested functions and annotated class fields. Imported names and dataclass-generated methods are not duplicated. Signatures and docstrings are extracted from source; undocumented behavior is explicitly identified rather than inferred from a name.

Usage links are statically resolved direct calls in package code and tests, capped at four examples per symbol. They are navigation aids, not an exhaustive runtime call graph; attribute calls, dynamic dispatch and external users may be absent. Private helpers are documented for review, not promised as public interfaces. Source line links describe this snapshot and may move as code changes.

| Module | Responsibility | Definitions |
| --- | --- | --- |
| [`__init__`](__init__.md) | Empty package initializer; no exports. | 0 |
| [`_ar1_moments`](_ar1_moments.md) | Lag-one residual autocorrelation and its projection-induced bias correction. | 18 |
| [`_arrays`](_arrays.md) | Owned NumPy arrays with protection against accidental writes. | 5 |
| [`_constants`](_constants.md) | Leaf home of the design constants; import them through ``_hrf_design``. | 0 |
| [`_conventional`](_conventional.md) | Internal support; inspect the definitions and their call sites below. | 24 |
| [`_denoising_cv`](_denoising_cv.md) | Leave-one-run-out scoring of temporal-PC counts and GLMsingle's pcstop rule. | 31 |
| [`_denoising_gate`](_denoising_gate.md) | Per-feature nested F-test significance gate on the pcstop PC count. | 19 |
| [`_denoising_identity`](_denoising_identity.md) | Content fingerprints that tie a denoising result to its analysis data. | 8 |
| [`_denoising_pool`](_denoising_pool.md) | GLMsingle ON-OFF R² noise pool, scoring features, and run-wise pool PCA. | 25 |
| [`_deprecation`](_deprecation.md) | One-release aliases for renamed keywords and functions. | 4 |
| [`_fit_diagnostics`](_fit_diagnostics.md) | Common diagnostics for conventional and prepared GLM fits. | 6 |
| [`_fit_lifecycle`](_fit_lifecycle.md) | Execution bookkeeping around explicit scientific fit operations. | 3 |
| [`_fractional_ridge`](_fractional_ridge.md) | Fractional ridge in the nuisance-projected, raw trial-coefficient basis. | 11 |
| [`_hrf_assignment`](_hrf_assignment.md) | Validate identified spatial assignments shared by GLM fitting workflows. | 1 |
| [`_hrf_cv`](_hrf_cv.md) | Nuisance-projected sufficient statistics and data-independent design caches. | 24 |
| [`_hrf_design`](_hrf_design.md) | One identified HRF per convolution; bases are never orthogonalized. | 17 |
| [`_hrf_glm`](_hrf_glm.md) | Fit conventional models in HRF groups and restore input feature order. | 12 |
| [`_hrf_glm_design`](_hrf_glm_design.md) | Compile conventional designs without changing semantic regressor names. | 9 |
| [`_mixture_threshold`](_mixture_threshold.md) | Two-component Gaussian-mixture tail threshold (GLMsingle's findtailthreshold). | 10 |
| [`_numerics`](_numerics.md) | Shared numerical policy: rank tolerance, column bases and projection error. | 7 |
| [`_reml_criterion`](_reml_criterion.md) | Profiled restricted likelihood for scalar REML. | 11 |
| [`_reml_inputs`](_reml_inputs.md) | Input validation for the public scalar REML entry points. | 8 |
| [`_reml_search`](_reml_search.md) | Vectorised search for the REML variance ratio. | 23 |
| [`_reml_terms`](_reml_terms.md) | Per-group sufficient statistics for scalar REML. | 12 |
| [`_resources.__init__`](_resources-__init__.md) | Empty package initializer; no exports. | 0 |
| [`_ridge_cv`](_ridge_cv.md) | Training-only HRF selection and streamed candidate beta-series prediction. | 11 |
| [`_scalars`](_scalars.md) | Scalar type predicates that never treat booleans as numbers. | 3 |
| [`_selected_hrf_fit`](_selected_hrf_fit.md) | Apply an identified HRF assignment, fitting only its feature groups. | 10 |
| [`_single_trial_design`](_single_trial_design.md) | Compile event-wise task columns without behavioral or condition pooling. | 6 |
| [`_single_trial_fit`](_single_trial_fit.md) | Run-wise OLS and normalized ridge with an unpenalized nuisance span. | 12 |
| [`_software`](_software.md) | Software versions for provenance, resolved only when requested. | 2 |
| [`_task_design`](_task_design.md) | Expand raw trials into Nilearn conditions and let Nilearn build the columns. | 8 |
| [`ar1`](ar1.md) | AR(1) whitening and bias-corrected noise autocorrelation estimates. | 15 |
| [`bids_provenance`](bids_provenance.md) | Project analysis records into BIDS metadata and provenance artifacts. | 37 |
| [`cifti`](cifti.md) | CIFTI-2 dense scalar export, checked reloads, and grayordinate identity. | 4 |
| [`cli`](cli.md) | Command line: `boldtailor run` executes the workflow on one BIDS subject's sessions. | 25 |
| [`data`](data.md) | Own and validate run signals, events, confounds, timing and provenance. | 26 |
| [`denoising`](denoising.md) | Opt-in task-guided denoising following GLMsingle's GLMdenoise stage. | 32 |
| [`denoising_results`](denoising_results.md) | Owned results of task-guided denoising selection. | 20 |
| [`design`](design.md) | Compile event-based task and nuisance design matrices. | 12 |
| [`diagnostics`](diagnostics.md) | Descriptive diagnostics of fitted trial betas: one-sample t and RT correlation. | 10 |
| [`fit`](fit.md) | Dispatch conventional fits and nested-OLS task variance comparisons. | 10 |
| [`fractional_ridge`](fractional_ridge.md) | Select relative shrinkage independently for each feature using encoding CV. | 3 |
| [`hrf_glm_results`](hrf_glm_results.md) | Conventional contrast results with spatially varying HRF designs. | 7 |
| [`hrf_library`](hrf_library.md) | Deterministic, peak-normalized HRFs; candidate 0 is Nilearn's SPM shape. | 51 |
| [`hrf_results`](hrf_results.md) | Owned results for HRF selection and independent evaluation. | 8 |
| [`hrf_selection`](hrf_selection.md) | Select HRFs by held-out task-model prediction across runs, without repeated images. | 13 |
| [`logging`](logging.md) | Bind fit context and collect structured lifecycle events. | 9 |
| [`model`](model.md) | Specify contrasts, nuisance settings and shared task regressors. | 41 |
| [`parallel`](parallel.md) | Bounded process batches; workers compute and the parent assembles outputs. | 3 |
| [`prepared`](prepared.md) | Own externally compiled designs and explicit column roles. | 31 |
| [`prepared_fit`](prepared_fit.md) | Fit externally compiled designs and compare task versus nuisance variance. | 17 |
| [`provenance`](provenance.md) | Identify source data and extend analysis records and fingerprints. | 45 |
| [`publication`](publication.md) | Stage and publish artifact sets with locking and rollback. | 28 |
| [`reliability`](reliability.md) | Descriptive agreement between selected HRF shapes, per grayordinate. | 9 |
| [`reml`](reml.md) | Scalar restricted maximum likelihood for one random-effect variance. | 27 |
| [`results`](results.md) | Own conventional contrast maps, R² diagnostics and result builders. | 44 |
| [`ridge_results`](ridge_results.md) | Owned numerical results for trial encoding and ridge selection. | 10 |
| [`ridge_selection`](ridge_selection.md) | Select one ridge penalty using held-out trial encoding across features. | 4 |
| [`single_trial`](single_trial.md) | Single-trial shared or selected HRF models without repeated-stimulus tuning. | 6 |
| [`single_trial_results`](single_trial_results.md) | Owned, read-only single-trial numerical results. | 10 |
| [`task_detection`](task_detection.md) | Detect a task model from event tables: response time and categorical trial type. | 9 |
| [`trial_encoding`](trial_encoding.md) | Predict trial betas with shared slopes and explicit run-baseline semantics. | 16 |
| [`workflow.__init__`](workflow-__init__.md) | Empty package initializer; no exports. | 0 |
| [`workflow.analysis`](workflow-analysis.md) | Blockwise public-API calls: HRF selection, GLMs, and beta series. | 14 |
| [`workflow.artifacts`](workflow-artifacts.md) | In-memory JSON, TSV, NPZ, PNG, and CIFTI artifacts for workflow outputs. | 7 |
| [`workflow.beta_series`](workflow-beta_series.md) | Fraction or shared-alpha tuning with independent odd/even evaluation. | 20 |
| [`workflow.files`](workflow-files.md) | BIDS/fMRIPrep discovery, CIFTI loading, nuisance selection, and sources. | 17 |
| [`workflow.inputs`](workflow-inputs.md) | Session loading, modulator detection, and the GLM model for any task. | 21 |
| [`workflow.outputs`](workflow-outputs.md) | Publish every workflow stage's maps, designs, tables, and provenance together. | 78 |
| [`workflow.plots`](workflow-plots.md) | Presentation for the workflow; inputs are already fitted arrays and tables. | 12 |
| [`workflow.report`](workflow-report.md) | Self-contained HTML report: one section per stage, figures embedded as PNG. | 24 |
| [`workflow.run`](workflow-run.md) | Run the enabled stages in order and publish one BIDS derivative with a report. | 30 |
| [`workflow.settings`](workflow-settings.md) | Resolved, validated settings for one workflow run; the CLI, stages, writer and metadata share it. | 34 |
| [`workflow.summaries`](workflow-summaries.md) | Small tables for the HTML report, built from results already in memory. | 8 |
| [`workflow.surfaces`](workflow-surfaces.md) | Offline cortical views of existing grayordinate statistics. | 6 |

Coverage: **76 modules; 1193 explicit definitions** (functions, classes and methods, including nested definitions).
