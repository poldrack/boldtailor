# Local publication migration

> **Superseded 2026-10-02.** Control data now lives in the sibling
> `<destination>.boldtailor/` directory (`lock`, `stage-<uuid>/`,
> `backup-<uuid>/`, `failures.jsonl`); payloads are not parsed and case-folded
> names are not scanned. See "Writing result files" in `development.md`.

Publication uses a local destination shared by cooperating writers. Artifact
paths are relative POSIX paths; existing symlinks inside the destination and
unrequested overwrites are rejected. Lock, staging, backups, and the failure
ledger live in the sibling control directory `<destination>.boldtailor/`, which
is not part of the dataset (no `.bidsignore` entry is needed). Protected
inputs (`source_paths`) must not overlap the outputs or this control directory.
The destination is resolved before checks, so aliases containing `..` cannot
bypass the source/output overlap check.

`lock_timeout` must be finite and non-negative; booleans are invalid.
Individual files are replaced atomically, but a set is not an atomic snapshot
for readers. Active directory swaps by other processes and crash recovery are
outside this contract.

## Signature change

Remove `retain_incomplete` from calls. The supported keyword arguments are
`source_paths`, `overwrite`, `lock_timeout`, and `keep_existing` (paths, such as
a shared dataset description, written only if absent once the lock is held). Successful rollback discards
staged outputs. Failed rollback automatically preserves its transaction;
there is no separate failed-artifact copy or rewritten provenance record.

## Recovering after a failed rollback

`PublicationError.__cause__` retains the original operation exception.
`rollback_errors` is a tuple of the recovery exceptions. When nonempty,
`recovery_directory` is an absolute `Path` to the retained
`<destination>.boldtailor/backup-<execution-id>/` directory, where originals
that could not be restored remain at `<artifact-path>`; when nothing had been
backed up, it names `stage-<execution-id>/` with the staged outputs instead.
The exception message also includes this location.

Stop other writers, inspect those backups and the destination, and resolve the
underlying filesystem problem before moving an original back. A destination
file may contain a partial new result; do not delete a backup merely because
its destination exists. No automatic recovery command is provided. Subsequent
writers leave retained transactions untouched.

Staging, backup, and promotion failures are caught as ordinary Python
exceptions. Rollback attempts every recorded restoration/removal, preserving
any failed originals. Successful publication and rollback clean up on a
best-effort basis; empty directories or scratch data may remain after cleanup
errors. Individual file fsync is retained without promising power-loss
recovery for the complete set. Process termination and fatal interruptions
can leave partial output and transaction files for manual inspection.

## Failure ledger fields

Records in `<destination>.boldtailor/failures.jsonl` use the `error_code`
categories of structured logs (`invalid_input`, `numerical_failure`,
`io_failure`, or `operation_failed`) instead of `error_type` and `message`.
Each record includes `execution_id`, `status`, `published`, and
`rollback_failed`; failed rollback also records `recovery_directory` as the
retained directory's name relative to the control directory. Caller-facing
exceptions retain absolute recovery paths and detailed causes. Ledgers written
by earlier releases inside the dataset (`.boldtailor/publication_failures.jsonl`)
are neither moved nor rewritten.

```json
{"error_code":"io_failure","execution_id":"123e4567-e89b-12d3-a456-426614174000","published":false,"recovery_directory":"backup-123e4567-e89b-12d3-a456-426614174000","rollback_failed":true,"status":"failed"}
```

The ledger is best effort: it cannot be guaranteed when its directory or file
is unwritable. Diagnostic-writing failure does not remove recovery backups or
replace the original exception. Records omit exception text and custom class
names; this is not a privacy guarantee for arbitrary artifact contents.
