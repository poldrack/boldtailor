"""Bounded process batches; workers compute and the parent assembles outputs."""

from itertools import islice
from numbers import Integral

from joblib import Parallel, delayed, parallel_config


def validate_n_jobs(n_jobs):
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    return int(n_jobs)


def execution_settings(n_jobs, block_size, n_features):
    workers = min(n_jobs, (n_features + block_size - 1) // block_size)
    return dict(
        n_jobs=n_jobs,
        active_workers=workers,
        block_size=block_size,
        backend="loky processes" if workers > 1 else "serial",
        worker_inner_threads=1 if workers > 1 else None,
    )


def map_blocks(function, blocks, *, args=(), n_jobs=1):
    """Yield ordered results, dispatching at most n_jobs blocks per batch."""
    n_jobs = validate_n_jobs(n_jobs)
    blocks = iter(blocks)
    if n_jobs == 1:
        for block in blocks:
            yield block, function(block, *args)
        return
    batch = list(islice(blocks, n_jobs))
    if len(batch) < 2:
        for block in batch:
            yield block, function(block, *args)
        return
    with parallel_config(backend="loky", inner_max_num_threads=1):
        with Parallel(
            n_jobs=len(batch), batch_size=1, pre_dispatch=len(batch), max_nbytes=None
        ) as parallel:
            while batch:
                results = parallel(delayed(function)(block, *args) for block in batch)
                yield from zip(batch, results, strict=True)
                # Release completed arrays before dispatching another batch.
                del results
                batch = list(islice(blocks, n_jobs))
