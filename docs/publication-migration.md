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
