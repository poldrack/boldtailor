# Local publication migration

Publication uses a local destination shared by cooperating writers. Artifact
paths are relative POSIX paths; existing symlinks and case-folded collisions
are rejected. The `.boldtailor` control directory is reserved regardless of
case. Destination aliases containing `..` are normalized after checking for
symlinks, so they cannot bypass source/output overlap checks.

`lock_timeout` must be finite and non-negative; booleans are invalid.
Individual files are replaced atomically, but a set is not an atomic snapshot
for readers. Active directory swaps by other processes and crash recovery are
outside this contract.

## Signature change

Remove `retain_incomplete` from calls. The supported keyword arguments are
`source_paths`, `overwrite`, and `lock_timeout`. Successful rollback discards
staged outputs. Failed rollback automatically preserves its transaction;
there is no separate failed-artifact copy or rewritten provenance record.

## Recovering after a failed rollback

`PublicationError.__cause__` retains the original operation exception.
`rollback_errors` is a tuple of the recovery exceptions. When nonempty,
`recovery_directory` is an absolute `Path` pointing to
`.boldtailor/transactions/<execution-id>/` under the destination. Originals
that could not be restored remain at `backups/<artifact-path>` there. The
exception message also includes this location.

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
