# Task 5 Real-Data and Repository Verification Report

## Status

**NEEDS_FIX**

The repository's synthetic verification, complete test suite, warning-strict suite,
formatting, lock, build, contract, initializer, diff, and status checks pass. The
required real-data notebook execution does not pass: the notebook selects
`ses-02` and `ses-04`, but the repaired dataset does not contain brain masks or
confounds for those sessions. The failure occurs during input discovery before
data loading, fitting, or publication.

No code, tests, notebook, dataset files, or dataset derivatives were changed. No
commit was created.

## Environment and scope

- Repository root:
  `/Users/poldrack/Dropbox/code/boldtailor/.worktrees/stop-signal-notebook`
- Read-only dataset:
  `/Users/poldrack/data_unsynced/rdoc_fmri`
- Prohibited persistent publication path:
  `/Users/poldrack/data_unsynced/rdoc_fmri/derivatives/boldtailor`
- Verification date: 2026-08-08
- The controller explicitly directed this worker not to dispatch another reviewer;
  final review remains controller-owned. This overrides Task 5 Step 4 for this
  worker.

## Command log and evidence

### 1. Brief and verification workflow read

Command:

```bash
sed -n '1,240p' /Users/poldrack/.codex/plugins/cache/claude-plugins-official/superpowers/6.2.0/skills/using-superpowers/SKILL.md && sed -n '1,280p' /Users/poldrack/.codex/plugins/cache/claude-plugins-official/superpowers/6.2.0/skills/verification-before-completion/SKILL.md && sed -n '1,320p' .superpowers/sdd/2026-08-08-stop-signal-notebook/task-5-brief.md
```

- Exit status: 0
- Relevant result: the Task 5 brief and verification requirements were loaded in
  full before repository verification.

### 2. Baseline repository status

Command:

```bash
uv run git status --short
```

- Exit status: 0
- Output: empty
- Result: the worktree was clean before verification.

### 3. Dataset and pre-execution publication-path check

Command:

```bash
uv run python -c 'from pathlib import Path; root=Path("/Users/poldrack/data_unsynced/rdoc_fmri"); target=root/"derivatives"/"boldtailor"; print(f"dataset_exists={root.is_dir()}"); print(f"publication_path={target}"); print(f"publication_exists_before={target.exists()}"); raise SystemExit(not root.is_dir() or target.exists())'
```

- Exit status: 0
- Output:

  ```text
  dataset_exists=True
  publication_path=/Users/poldrack/data_unsynced/rdoc_fmri/derivatives/boldtailor
  publication_exists_before=False
  ```

### 4. Required real-data notebook execution

Command (the brief's required execution, wrapped only with `/usr/bin/time -p` to
capture timing):

```bash
/usr/bin/time -p env BOLDTAILOR_BIDS_ROOT=/Users/poldrack/data_unsynced/rdoc_fmri MPLBACKEND=Agg uv run python -c 'from pathlib import Path; import nbformat; from nbclient import NotebookClient; path=Path("examples/stop_signal_demo.ipynb"); notebook=nbformat.read(path, as_version=4); NotebookClient(notebook, timeout=900, kernel_name="python3", resources={"metadata":{"path":str(Path.cwd())}}).execute(); print("real notebook execution passed")'
```

First sandboxed attempt:

- Exit status: 2
- Timing: real 0.01 s, user 0.00 s, sys 0.00 s
- Output:

  ```text
  error: failed to open file `/Users/poldrack/.cache/uv/sdists-v9/.git`: Operation not permitted (os error 1)
  real 0.01
  user 0.00
  sys 0.00
  ```

The same command was rerun after approval with access to the existing uv cache and
the explicitly scoped read-only dataset.

Approved execution:

- Exit status: 1
- Timing: real 2.90 s, user 1.91 s, sys 0.41 s
- Result: failed in notebook cell 3 while constructing `inputs`, before data
  loading, fitting, or publication.
- Exact terminal exception:

  ```text
  FileNotFoundError: mask discovery expected exactly one file

  real 2.90
  user 1.91
  sys 0.41
  ```

- The trace identifies
  `examples/stop_signal_demo.py:65` in `discover_run_inputs()`, where discovery
  searches for
  `{stem}_space-{space}_res-{resolution}_desc-brain_mask.nii.gz`, and `_one()`
  raises at line 81 because it receives no matches.
- Expected output `real notebook execution passed` was not printed.
- A Jupyter transport warning preceded the exception; it was not the cause of the
  failure.

### 5. Immediate post-failure publication-path check

Command:

```bash
uv run python -c 'from pathlib import Path; target=Path("/Users/poldrack/data_unsynced/rdoc_fmri/derivatives/boldtailor"); print(f"publication_exists_after_failed_execution={target.exists()}"); raise SystemExit(target.exists())'
```

- Exit status: 0
- Output: `publication_exists_after_failed_execution=False`
- Result: no persistent dataset derivative was created.

### 6. Notebook configuration and derivative inventory

The following read-only inspection commands established that this is a real
notebook/data-selection mismatch rather than an ambiguous glob.

Command:

```bash
uv run python -c 'from pathlib import Path; import nbformat; p=Path("examples/stop_signal_demo.ipynb"); nb=nbformat.read(p, as_version=4); print("NOTEBOOK CONFIG/CODE"); print("\n".join(nb.cells[0].source.splitlines())); print("\nDATASET FUNCTIONAL FILES"); root=Path("/Users/poldrack/data_unsynced/rdoc_fmri/derivatives/fmriprep"); files=sorted(root.glob("sub-*/ses-*/func/*")); print("\n".join(str(x.relative_to(root)) for x in files))'
```

- Exit status: 0
- Relevant result: cell 0 is Markdown and the probed legacy `derivatives/fmriprep`
  path contains no functional files. This prompted inspection of the notebook's
  actual configured derivative root.

Command:

```bash
uv run python -c 'from pathlib import Path; import nbformat; nb=nbformat.read("examples/stop_signal_demo.ipynb", as_version=4); print("NOTEBOOK CELLS WITH CONFIG"); print("\n---\n".join(f"CELL {i}\n{c.source}" for i,c in enumerate(nb.cells) if any(x in c.source for x in ("BIDS_ROOT", "FMRIPREP_ROOT", "SUBJECT", "SESSIONS", "RESOLUTION")))); print("\nDERIVATIVE DIRECTORIES"); root=Path("/Users/poldrack/data_unsynced/rdoc_fmri/derivatives"); print("\n".join(str(x.relative_to(root)) for x in sorted(root.glob("*")))); print("\nMASK FILES"); print("\n".join(str(x) for x in sorted(root.rglob("*mask*.nii*"))))'
```

- Exit status: 0
- Notebook configuration evidence:

  ```text
  FMRIPREP_ROOT = BIDS_ROOT / "derivatives" / "fmri_25.2.0"
  SUBJECT = "sub-s4"
  SESSIONS = ("ses-02", "ses-04")
  TASK = "stopSignal"
  RUN = "run-01"
  SPACE = "MNI152NLin2009cAsym"
  RESOLUTION = 2
  PERSIST_DERIVATIVES = False
  ```

- Dataset evidence: the only derivative directory is `fmri_25.2.0`. Mask files
  exist for `ses-06`, `ses-08`, and `ses-10`; no mask files exist for `ses-02`
  or `ses-04`.

Command:

```bash
uv run python -c 'from pathlib import Path; root=Path("/Users/poldrack/data_unsynced/rdoc_fmri"); print("RAW EVENT FILES"); print("\n".join(str(x.relative_to(root)) for x in sorted(root.glob("sub-s4/ses-*/func/*events.tsv")))); print("\nDERIVATIVE BOLD FILES"); print("\n".join(str(x.relative_to(root)) for x in sorted((root/"derivatives"/"fmri_25.2.0").glob("sub-s4/ses-*/func/*space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"))))'
```

- Exit status: 0
- Relevant result: raw stop-signal event files and matching preprocessed BOLD
  files exist for `ses-02`, `ses-04`, `ses-06`, `ses-08`, and `ses-10` (with
  `ses-10` using `run-02`). BOLD availability alone does not make `ses-02` and
  `ses-04` complete notebook inputs.

Command:

```bash
uv run python -c 'from pathlib import Path; root=Path("/Users/poldrack/data_unsynced/rdoc_fmri/derivatives/fmri_25.2.0/sub-s4");
for session in ("ses-02", "ses-04"):
 print(f"[{session}]"); print("\n".join(x.name for x in sorted((root/session/"func").glob("*"))))'
```

- Exit status: 0
- Exact `ses-02` inventory:

  ```text
  sub-s4_ses-02_task-stopSignal_run-01_hemi-L_space-fsnative_bold.func.gii
  sub-s4_ses-02_task-stopSignal_run-01_hemi-R_space-fsnative_bold.func.gii
  sub-s4_ses-02_task-stopSignal_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
  ```

- Exact `ses-04` inventory:

  ```text
  sub-s4_ses-04_task-stopSignal_run-01_hemi-L_space-fsnative_bold.func.gii
  sub-s4_ses-04_task-stopSignal_run-01_hemi-R_space-fsnative_bold.func.gii
  sub-s4_ses-04_task-stopSignal_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz
  ```

- Result: neither configured session contains the required volumetric brain mask
  or confounds TSV.

Command:

```bash
uv run python -c 'from pathlib import Path; root=Path("/Users/poldrack/data_unsynced/rdoc_fmri"); print("CONFOUNDS"); print("\n".join(str(x.relative_to(root)) for x in sorted(root.rglob("sub-s4_ses-*_task-stopSignal_run-*_desc-confounds_timeseries.tsv"))))'
```

- Exit status: 0
- Output:

  ```text
  derivatives/fmri_25.2.0/sub-s4/ses-06/func/sub-s4_ses-06_task-stopSignal_run-01_desc-confounds_timeseries.tsv
  derivatives/fmri_25.2.0/sub-s4/ses-08/func/sub-s4_ses-08_task-stopSignal_run-01_desc-confounds_timeseries.tsv
  derivatives/fmri_25.2.0/sub-s4/ses-10/func/sub-s4_ses-10_task-stopSignal_run-02_desc-confounds_timeseries.tsv
  ```

- Result: complete `run-01` BOLD/mask/confounds derivative sets exist for
  `ses-06` and `ses-08`; the configured `ses-02` and `ses-04` do not contain
  masks or confounds.

The inventory and defect were reported to the controller before any code change.
The controller independently confirmed the inventory and directed this worker to
make no changes and return `NEEDS_FIX`.

### 7. Focused tests

Command:

```bash
uv run pytest tests/test_stop_signal_demo.py -q
```

- Exit status: 0
- Output: `18 passed in 4.13s`

### 8. Complete test suite

Command:

```bash
uv run pytest -q
```

- Exit status: 0
- Output: `205 passed in 6.08s`

### 9. Warning-strict complete test suite

Command:

```bash
uv run pytest -q -W error
```

- Exit status: 0
- Output: `205 passed in 5.92s`
- Result: no warning was promoted to a failure.

### 10. Formatting

Command:

```bash
uv run black --check src tests examples/stop_signal_demo.py
```

- Exit status: 0
- Output: `24 files would be left unchanged.`

### 11. Dependency lock

Command:

```bash
uv lock --check
```

- Exit status: 0
- Output: `Resolved 76 packages in 4ms`

### 12. Package build

Command:

```bash
uv build
```

- Exit status: 0
- Relevant output:

  ```text
  Successfully built dist/boldtailor-0.1.0.tar.gz
  Successfully built dist/boldtailor-0.1.0-py3-none-any.whl
  ```

### 13. Required build-artifact cleanup

Command:

```bash
uv run rm -rf dist src/boldtailor.egg-info
```

- Exit status: 0
- Output: empty
- Result: only artifacts generated by the immediately preceding build were
  removed.

### 14. Repository contract tests

Command:

```bash
uv run pytest tests/test_repository_contracts.py -q
```

- Exit status: 0
- Output: `2 passed in 0.00s`

### 15. Empty initializer contract

Command:

```bash
uv run python -c 'from pathlib import Path; files=list(Path("src").rglob("__init__.py")); bad=[str(path) for path in files if path.read_bytes()]; print(f"checked {len(files)} initializers"); raise SystemExit(bool(bad))'
```

- Exit status: 0
- Output: `checked 1 initializers`
- Result: all discovered initializers are empty.

### 16. Initial diff check

Command:

```bash
uv run git diff --check
```

- Exit status: 0
- Output: empty

### 17. Initial post-test status check

Command:

```bash
uv run git status --short
```

- Exit status: 0
- Output:

  ```text
  ?? examples/__pycache__/
  ?? src/boldtailor/__pycache__/
  ?? tests/__pycache__/
  ```

- Result: these were exact cache artifacts created by this verification; no code,
  notebook, build distribution, or dataset artifact was present.

### 18. Exact generated-artifact existence check

Command:

```bash
uv run python -c 'from pathlib import Path; paths=[Path("build"), Path("examples/__pycache__"), Path("src/boldtailor/__pycache__"), Path("tests/__pycache__")]; print("\n".join(f"{p}: exists={p.exists()}" for p in paths))'
```

- Exit status: 0
- Output:

  ```text
  build: exists=False
  examples/__pycache__: exists=True
  src/boldtailor/__pycache__: exists=True
  tests/__pycache__: exists=True
  ```

### 19. Exact cache cleanup

Command:

```bash
uv run rm -rf examples/__pycache__ src/boldtailor/__pycache__ tests/__pycache__
```

- Exit status: 0
- Output: empty
- Result: only the three cache directories generated during this verification were
  removed.

### 20. Fresh final diff check

Command:

```bash
uv run git diff --check
```

- Exit status: 0
- Output: empty

### 21. Fresh final repository status

Command:

```bash
uv run git status --short
```

- Exit status: 0
- Output: empty
- Result: no generated build, notebook-execution, cache, code, test, or derivative
  artifacts remain in the worktree.

### 22. Fresh final publication-path check

Command:

```bash
uv run python -c 'from pathlib import Path; target=Path("/Users/poldrack/data_unsynced/rdoc_fmri/derivatives/boldtailor"); print(f"publication_exists_final={target.exists()}"); raise SystemExit(target.exists())'
```

- Exit status: 0
- Output: `publication_exists_final=False`
- Result: the dataset's prohibited persistent publication path was not created.

## Verification summary

| Check | Result |
|---|---:|
| Real-data notebook execution | **FAIL** — missing mask for configured session |
| Persistent dataset publication absent before execution | PASS |
| Persistent dataset publication absent after execution | PASS |
| Focused stop-signal tests | PASS — 18/18 |
| Complete tests | PASS — 205/205 |
| Warning-strict complete tests | PASS — 205/205 |
| Black formatting | PASS — 24 files unchanged |
| uv lock check | PASS — 76 packages resolved |
| Source and wheel build | PASS |
| Repository contract tests | PASS — 2/2 |
| Empty initializers | PASS — 1/1 |
| Diff check | PASS |
| Final status | PASS — clean |
| Dataset changed | NO |
| Code/tests/notebook changed | NO |
| Commits created | NONE |

## Concerns and required follow-up

1. The notebook cannot execute against the brief's repaired real dataset while it
   selects `ses-02` and `ses-04`; those sessions lack both required brain masks and
   confounds. This blocks the required real execution evidence and prevents runtime
   verification of temporary publication output.
2. `ses-06` and `ses-08` have complete `run-01` BOLD/mask/confounds derivative
   sets and matching raw event files. Selecting them is the apparent repair, but no
   implementation or test change was authorized in this verification-only task.
3. Because execution stops during discovery, the notebook's runtime output cannot
   yet confirm the temporary destination basename, artifact count, or published
   relative paths. Static configuration does set `PERSIST_DERIVATIVES = False`, and
   all before/after/final filesystem checks confirm that
   `derivatives/boldtailor` was never created.
4. The final code review requested by Task 5 Step 4 was not dispatched by this
   worker because the controller explicitly retained ownership of final review.

## Fix round: configurable complete-session defaults

### Status

**PASS**

The notebook now defaults to the complete real-data sessions `ses-06` and
`ses-08`. `BOLDTAILOR_SESSIONS` accepts exactly two non-empty comma-separated
session labels after trimming surrounding whitespace, and the synthetic notebook
smoke test explicitly selects its `ses-02,ses-04` fixture through that override.

### RED commit and failure evidence

- Commit:
  `12ccb63b25be6ce29a197110db0502d34a622e77`
- Commit subject: `test: specify configurable notebook sessions`
- Tests-only changed file: `tests/test_stop_signal_demo.py`
- Exact command:

  ```bash
  uv run pytest tests/test_stop_signal_demo.py -q
  ```

- Exit status: 1
- Result: `5 failed, 19 passed in 4.24s`
- Expected failure reasons: executing the notebook configuration returned
  `('ses-02', 'ses-04')` instead of `('ses-06', 'ses-08')`, and each malformed
  override case failed with `DID NOT RAISE ValueError` because the notebook did not
  yet read or validate `BOLDTAILOR_SESSIONS`.

### GREEN commit and passing evidence

- Commit:
  `64da94873834afedde99802f7974fb531212e3db`
- Commit subject: `fix: select complete stop-signal sessions`
- Exact focused command:

  ```bash
  uv run pytest tests/test_stop_signal_demo.py -q
  ```

  Result: exit 0, `24 passed in 4.18s`.

- Exact complete-suite command:

  ```bash
  uv run pytest -q
  ```

  Result: exit 0, `211 passed in 6.05s`.

- Exact warning-strict command:

  ```bash
  uv run pytest -q -W error
  ```

  Result: exit 0, `211 passed in 5.97s`.

- Exact real-data command, with no `BOLDTAILOR_SESSIONS` override:

  ```bash
  BOLDTAILOR_BIDS_ROOT=/Users/poldrack/data_unsynced/rdoc_fmri \
  MPLBACKEND=Agg \
  uv run python -c 'from pathlib import Path; import nbformat; from nbclient import NotebookClient; path=Path("examples/stop_signal_demo.ipynb"); notebook=nbformat.read(path, as_version=4); NotebookClient(notebook, timeout=900, kernel_name="python3", resources={"metadata":{"path":str(Path.cwd())}}).execute(); print("real notebook execution passed")'
  ```

  Result: exit 0 and `real notebook execution passed`. The kernel emitted its
  standard unencrypted-TCP transport warning; execution itself completed.

- Additional exact checks and results:

  - `uv run black --check src tests examples/stop_signal_demo.py` — exit 0,
    `24 files would be left unchanged.`
  - `uv lock --check` — exit 0, `Resolved 76 packages in 3ms`.
  - `uv build` — exit 0; source distribution and wheel built successfully, then
    all generated build artifacts were removed.
  - `uv run pytest tests/test_repository_contracts.py -q` — exit 0,
    `2 passed in 0.00s`.
  - `uv run python -c 'from pathlib import Path; files=list(Path("src").rglob("__init__.py")); bad=[str(path) for path in files if path.read_bytes()]; print(f"checked {len(files)} initializers; nonempty={bad}"); raise SystemExit(bool(bad))'`
    — exit 0, `checked 1 initializers; nonempty=[]`.
  - `uv run python -c 'import json; from pathlib import Path; json.loads(Path("examples/stop_signal_demo.ipynb").read_text()); print("notebook JSON valid")'`
    — exit 0, `notebook JSON valid`.
  - `uv run git diff --check` — exit 0 with no output.

### Changed files

- `tests/test_stop_signal_demo.py` — behaviorally executes the configuration cell;
  covers the `ses-06,ses-08` default, normalized fixture override, invalid override
  rejection, and supplies the override to the smoke execution.
- `examples/stop_signal_demo.ipynb` — adds the validated environment override and
  complete-session default; removes the old dataset-specific design inequality.
- `docs/superpowers/specs/2026-08-08-stop-signal-notebook-design.md` — documents the
  corrected real defaults and synthetic override.
- `docs/superpowers/plans/2026-08-08-stop-signal-notebook.md` — corrects the plan's
  real defaults, configuration example, smoke setup, and design-matrix narrative.
- `.superpowers/sdd/2026-08-08-stop-signal-notebook/task-5-report.md` — records this
  fix round and its immutable RED/GREEN evidence.

### Self-review

- Tests exercise the executed notebook configuration rather than matching exact
  notebook source text.
- The override parser is confined to a short configuration-cell function, trims
  values, preserves tuple semantics, and enforces the existing two-session
  contract without expanding the example helper API.
- The old assertion that the two designs must differ was incompatible with the
  complete default sessions, whose modeled condition sets are the same; the
  notebook still displays and plots each run-specific design.
- No package initializer or installed API changed. Notebook execution did not save
  outputs back into the repository.

### External-data and derivative safety

The external dataset was read only. No command modified
`/Users/poldrack/data_unsynced/rdoc_fmri`. The persistent derivative path
`/Users/poldrack/data_unsynced/rdoc_fmri/derivatives/boldtailor` was absent both
immediately before and immediately after the successful real-data execution. The
notebook retained temporary publication as its default, and all repository build
and cache artifacts created during verification were removed.
