"""Bounded process batches; workers compute and the parent assembles outputs.

Workers must return picklable arrays or dictionaries, not result objects.
"""

from itertools import islice
from numbers import Integral

from joblib import Parallel, delayed, parallel_config


def validate_n_jobs(n_jobs):
    """Return a positive integer worker count; booleans are rejected."""
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    return int(n_jobs)


def _serial(function, blocks, args):
    for block in blocks:
        yield block, function(block, *args)


def map_blocks(function, blocks, *, args=(), n_jobs=1):
    """Yield ordered ``(block, result)`` pairs, dispatching at most n_jobs blocks.

    Several workers run in loky processes with one inner BLAS/OpenMP thread;
    each batch is released before the next one is dispatched.
    """
    n_jobs = validate_n_jobs(n_jobs)
    blocks = iter(blocks)
    batch = list(islice(blocks, n_jobs)) if n_jobs > 1 else []
    if n_jobs == 1 or len(batch) < 2:
        yield from _serial(function, batch or blocks, args)
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
