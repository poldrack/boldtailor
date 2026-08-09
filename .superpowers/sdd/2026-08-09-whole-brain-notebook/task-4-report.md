# Task 4 Report: Real-Data and Repository Verification

## Scope

- Verified head: `d366b995add997d1ab6578f50c57ba0ba2239699`
- Dataset: `/Users/poldrack/data_unsynced/rdoc_fmri`, read-only
- Notebook publication mode: temporary default
- No source, test, notebook, README, lock, or project-metadata file changed

## Persistent-destination guards

- Before execution: `publication_exists_before=False`, exit 0.
- After execution: `publication_exists_after=False`, exit 0.
- Exact guarded path: `/Users/poldrack/data_unsynced/rdoc_fmri/derivatives/boldtailor`.

The real dataset was not modified and no persistent Boldtailor derivative was created.

## Real two-session execution

The notebook was executed from `examples/` with:

- `BOLDTAILOR_BIDS_ROOT=/Users/poldrack/data_unsynced/rdoc_fmri`
- `MPLBACKEND=Agg`
- task-specific IPython and Jupyter directories under `/private/tmp`
- `NotebookClient(..., timeout=7200, kernel_name="python3")`

Result: exit 0 in 23.87 seconds. The rendered-output audit returned
`{'missing': []}` for all required values:

- `ses-06`
- `ses-08`
- `successful_inhibition`
- `stop_vs_go`
- `go_success_vs_baseline`
- `common_voxel_count`
- `estimated_signal_memory_gib`
- `published_count`

The temporary publication was cleaned up by the notebook-owned temporary
directory lifecycle.

## Repository gates

- Focused warning-strict suite: `32 passed in 11.92s`.
- Full warning-strict suite: `219 passed in 13.53s`.
- Black: `24 files would be left unchanged`.
- Lock: `Resolved 76 packages in 9ms`.
- Build: source distribution and wheel built successfully under
  `/private/tmp/boldtailor-task4-dist`.
- Repository contracts: `2 passed in 0.00s`.
- Initializers: `checked 1 initializers; nonempty=[]`.
- `git diff --check`: exit 0.

The build created `src/boldtailor.egg-info`, which was confirmed absent before
the build and removed afterward. It is regenerable with `uv build`. The build
did not leave `build/` or `dist/` in the worktree.

## Final status and concerns

The only untracked paths are the same pre-existing disposable caches recorded
before Task 4:

```text
?? examples/__pycache__/
?? src/boldtailor/__pycache__/
?? tests/__pycache__/
```

They were not deleted because they predated this verification. No functional
or repository-gate concern remains. The shell prints a benign `VIRTUAL_ENV`
selection notice because the parent checkout and isolated worktree have
different virtual environments; uv selected the worktree environment.
