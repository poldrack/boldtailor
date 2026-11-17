# Workflow/CLI verification ledger (2026-10-03)

## Test counts

| Suite | Before (0eca362) | After first review | After final review fixes |
| --- | --- | --- | --- |
| default (`uv run pytest -q -W error`) | 1079 passed, 1 skipped | 1255 passed, 1 skipped | 1307 passed, 1 skipped |
| `examples/NSD` | not recorded | 43 passed, 3 skipped | 43 passed, 3 skipped |
| `examples/validation` | not recorded | 24 passed | not rerun |

## Checks

- `uv run black --check src tests examples`: pass (152 files unchanged, after formatting `tests/test_docs_mention_cli.py`).
- `git diff --check`: pass.
- `grep -rn "examples" src/boldtailor`: no hits.
- `grep -rn "desc-notebook" src examples docs`: no hits after fixing `docs/development.md:299` to `<stem>_desc-<GLM descriptor>_designs.npz` (matches `_design_artifact` in `workflow/outputs.py`). Excluded: `docs/superpowers/` and `docs/review-*` (historical records).
- `grep -rn "center=" src tests examples docs`: one hit, `tests/test_task_design.py:76` (asserts `Modulator(..., center=False)` raises TypeError; Ruling R11). Excluded: `examples/validation` (unrelated plotting arguments), `docs/superpowers/`, and `docs/review-2026-10-01-full-project.md` (historical review text).
- `uv run boldtailor run --help`: exit 0.
- CLI dry run on the four-run synthetic dataset (`--dry-run`): exit 0, JSON printed.
- Real run (`--hrf-library canonical --ridge-mode off --n-jobs 1 --block-size 2 --no-surface-maps --output-dir <tmp>/out`): exit 0, 66 files written, `sub-07_ses-nsd10_task-nsdcore_report.html` exists.
- R9 check (four runs, default ridge mode, no `--dry-run`): exit 2, one stderr line: `input error: ridge cross-validation needs at least three odd and three even runs; use --ridge-mode off|fixed or --skip-stage betas`; no output directory created.

## Final-review checks

- `uv run black --check src tests examples`: pass (153 files unchanged).
- `uv run boldtailor run --help`: exit 0; lists `--no-modulators` and `--quiet`.
- Dry run, four-run synthetic dataset, `--hrf-library canonical --ridge-mode off --no-surface-maps`: exit 0, JSON with `notes` and empty `problems`.
- Dry run with the default ridge mode on four runs: exit 2, JSON printed, `input error: ridge cross-validation needs at least three odd and three even runs; ...` on stderr (M4: the dry run exits as the real run would).
- Real run (`--hrf-library canonical --ridge-mode off --no-surface-maps --n-jobs 1 --block-size 2`): exit 0, 66 files, report written; timestamped progress lines on stderr. A second run exits 1 (`error: existing files in ... (<up to five names>, ... N files); set existing_results to overwrite ...`); `--existing-results overwrite --quiet` exits 0 with no stderr output.
- `--output-dir <bids>/derivatives/fmriprep --existing-results overwrite`: exit 1 (`output_dir ... must not be bids_dir, nor be, contain, or lie inside the fMRIPrep directory`); every input dtseries intact.
- String `trial_type` (face/house), no `response_time`, dry run with `--ridge-mode off`: exit 0, regressors `["task"]`, note `trial_type is not binary 0/1; not used as a modulator`. With the default ridge mode: exit 2 (parity, R9; a six-run task-only session gets the R13 message).

## Rulings
- Ruling R1: HRF artifacts from the `glms` stage are the all-runs selection only (`HRF`/`HRFAll` maps, library table, provenance); odd/even split artifacts (`HRFOdd`, `HRFEven`, `HRFOddToEven`, `HRFEvenToOdd`, `HRFReliability`) are written only when the reliability stage ran — spec §Stage pipeline puts odd/even selections in `reliability`, and the T11 test requires it — cost if wrong: multisession readers that expect HRFOdd/HRFEven from glms-only runs miss them.
- Ruling R2: in the CLI, `--space`, `--hrf-library`, `--ridge-mode`, `--encoding-mode`, `--existing-results` take plain strings (allowed values listed in help/metavar) and are validated by WorkflowSettings (exit 1, one stderr line); the parser's `error()` prints one line and exits 1 (spec: "1 usage or settings error (one line on stderr)"); `--skip-stage` keeps `choices` so `glms` is rejected; `main` returns the parser's exit code instead of raising — spec exit-code contract plus T13 tests — cost if wrong: weaker argparse-level help for invalid choices.
- Ruling R3: BetaModel uses the plan's fields (`name`, `hrf`, `estimator`, `fit`, `tuning`, `evaluation`, `cv_provenance`, `predictors`); Task 1 adds a fourth spec amendment recording this — plan is the more detailed argument and later tasks build on it — cost if wrong: spec/plan naming mismatch.
- Ruling R4: commit trailers use `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` (harness attribution for this session) instead of the plan's Fable 5.1 line — the plan line reflects the model that authored the plan — cost if wrong: cosmetic trailer text.
- Ruling R5: existing commits whose trailers name the model that actually wrote them (e.g. Haiku 4.5) stand; no history rewrite (another implementer works on the branch, and the trailer is accurate attribution) — cost if wrong: inconsistent trailers across commits.
- Ruling R6: the opt-in examples/NSD suite may be red between Tasks 4 and 14 (each task git-mv's a module whose example call sites change signature; Task 14 re-points and restores them); each move task must keep the default suite green and must not leave examples importing a deleted module where a one-line import re-point suffices — the plan sequences it so and Task 14/16 verify the examples suite — cost if wrong: intermediate commits have broken opt-in example tests.
- Ruling R7: `existing_results="overwrite"` replaces the subject/session/task output set: after (or inside) the publication transaction, files matching `<sub>_<ses>_task-<task>_*` in the func dir that are not in the new artifact set are removed — spec §Outputs says overwrite "replaces the subject/session outputs in one publication transaction" — cost if wrong: an overwrite run deletes stale files a user expected to keep (only within that sub/ses/task prefix).
- Ruling R8: Task 10's report includes the spec's "equivalent command line" in the settings section and one-line notes for disabled/skipped stages (spec §HTML report), beyond the plan's snippet — cost if wrong: extra report code.
- Ruling R9: when betas is enabled with ridge_mode cv/fractional_cv and either parity half has fewer than two runs, run_workflow raises InputError (CLI exit 2) right after loading and before any fitting, naming the requirement and suggesting --ridge-mode off|fixed or --skip-stage betas — failing late after GLM fitting wastes compute and writes nothing; silently dropping CV estimators would change the requested outputs — cost if wrong: users with two-run sessions must pass a flag instead of getting OLS-only betas automatically.
- Ruling R9 (amended): optimized-HRF ridge CV needs at least three odd and three even runs (beta_series.py minimum = 3 with a library), and fit_beta_models always fits it in cv/fractional_cv, so the fail-fast check requires >=3 runs per parity half in those modes — found by Task 14 review — cost if wrong: 4-5 run sessions must pass --ridge-mode off|fixed.
- Ruling R10: in the CLI, skipping a stage also drops the stages that require it (--skip-stage betas ⇒ summaries skipped), while WorkflowSettings stays strict — docs and the R9 error hint promise it, and an exit-1 for '--skip-stage betas' is hostile — cost if wrong: summaries silently dropped when betas is skipped (documented). Sent to Task 11 implementer with the R9 amendment.
- Ruling R11: the one remaining 'center=' hit (tests/test_task_design.py, asserting Modulator(center=...) raises TypeError) is plan-mandated by Task 2 and satisfies Task 16's grep intent — cost if wrong: none.
- Ruling R12: trial_type is auto-detected only when every run's values are numeric 0/1; otherwise it is omitted with a recorded note (dry run, metadata `notes`, report inputs section); `--no-modulators` gives an explicit task-only model (`modulators=()`, round-trips through `report.command_line`); explicit `--modulator trial_type` keeps the strict 0/1 check, naming the BIDS run — BIDS trial_type is normally a string label and the spec's goal is non-NSD use — cost if wrong: a dataset with a non-binary numeric trial_type silently loses that modulator (noted in the report). USER SHOULD CONFIRM.
- Ruling R13: ridge cv/fractional_cv with a task model that has no modulators fails fast with InputError before fitting (same place as R9), suggesting `--ridge-mode off|fixed` or `--skip-stage betas` — consistent with R9 — cost if wrong: task-only users must pass a flag.
- Ruling R14 (narrows R7): WorkflowSettings rejects an output_dir equal to bids_dir or equal to, inside, or containing fmriprep_dir (resolved paths); every run records its published paths in `desc-boldtailor_metadata.json` under `artifacts`, and an overwrite deletes only stale files from the previous list (none without a list), never a run source and never a path outside output_dir — C1 data-loss risk — cost if wrong: an older output without an artifact list keeps stale files after overwrite.
