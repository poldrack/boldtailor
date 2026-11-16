"""Bounded process batches; workers compute and the parent assembles outputs.

Result objects that hold rebuild closures (``SingleTrialResult`` with a
``SelectedTrialDesign``, ``HrfAnalysisResult``) cannot be serialized with the
standard ``pickle`` module; loky's cloudpickle can move them between
processes, but workers should prefer returning arrays or dicts to keep
transfers small.
"""

from itertools import islice
from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar, cast
from joblib import Parallel, delayed, parallel_config

from boldtailor._scalars import is_integer


def validate_n_jobs(n_jobs):
    """Return a positive integer worker count; booleans are rejected."""
    if not is_integer(n_jobs) or n_jobs < 1:
        raise ValueError("n_jobs must be a positive integer")
    return int(n_jobs)


_Block = TypeVar("_Block")
_Result = TypeVar("_Result")


def _serial(
    function: Callable[..., _Result], blocks: Iterable[_Block], args: tuple[object, ...]
) -> Iterator[tuple[_Block, _Result]]:
    for block in blocks:
        yield block, function(block, *args)


def map_blocks(
    function: Callable[..., _Result],
    blocks: Iterable[_Block],
    *,
    args: tuple[object, ...] = (),
    n_jobs: int = 1,
) -> Iterator[tuple[_Block, _Result]]:
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
        # Joblib accepts integer batching; its inferred types cover string defaults.
        with Parallel(
            n_jobs=len(batch),
            batch_size=1,  # pyright: ignore[reportArgumentType]
            pre_dispatch=len(batch),  # pyright: ignore[reportArgumentType]
            max_nbytes=None,
        ) as parallel:
            while batch:
                # Joblib's default return_as="list" preserves the worker's values.
                results = cast(
                    list[_Result],
                    parallel(delayed(function)(block, *args) for block in batch),
                )
                yield from zip(batch, results, strict=True)
                # Release completed arrays before dispatching another batch.
                del results
                batch = list(islice(blocks, n_jobs))
