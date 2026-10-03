"""Execution records for the NSD scripts; batching lives in boldtailor.parallel."""


def execution_settings(n_jobs, block_size, n_features):
    workers = min(n_jobs, (n_features + block_size - 1) // block_size)
    return dict(
        n_jobs=n_jobs,
        active_workers=workers,
        block_size=block_size,
        backend="loky processes" if workers > 1 else "serial",
        worker_inner_threads=1 if workers > 1 else None,
    )
