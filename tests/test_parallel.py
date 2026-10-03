"""Process batches stay bounded, ordered, and single-threaded per worker."""

import os
import time

import pytest

from boldtailor.parallel import map_blocks, validate_n_jobs


def _process_probe(block, directory):
    (directory / str(block)).write_text(str(os.getpid()))
    if block < 2:
        deadline = time.monotonic() + 15
        while not all((directory / str(i)).exists() for i in range(2)):
            if time.monotonic() > deadline:
                raise RuntimeError("two blocks did not execute concurrently")
            time.sleep(0.01)
    if block == 0:
        time.sleep(0.1)
    return dict(
        pid=os.getpid(),
        openmp=os.environ.get("OMP_NUM_THREADS"),
        accelerate=os.environ.get("VECLIB_MAXIMUM_THREADS"),
    )


def test_process_batches_are_bounded_ordered_and_thread_limited(tmp_path):
    results = map_blocks(_process_probe, range(4), args=(tmp_path,), n_jobs=2)
    first = next(results)
    assert first[0] == 0
    assert {p.name for p in tmp_path.iterdir()} == {"0", "1"}
    all_results = [first, *results]
    assert [block for block, _ in all_results] == [0, 1, 2, 3]
    assert all(result["pid"] != os.getpid() for _, result in all_results)
    assert len({result["pid"] for _, result in all_results[:2]}) == 2
    assert all(
        result["openmp"] == result["accelerate"] == "1" for _, result in all_results
    )


def test_serial_execution_stays_in_process_and_ordered():
    results = list(map_blocks(lambda block, offset: block + offset, [3, 1], args=(10,)))
    assert results == [(3, 13), (1, 11)]


@pytest.mark.parametrize("n_jobs", [0, -1, True, 1.5])
def test_invalid_worker_count_is_rejected(n_jobs):
    with pytest.raises(ValueError, match="n_jobs.*positive integer"):
        validate_n_jobs(n_jobs)
    with pytest.raises(ValueError, match="n_jobs.*positive integer"):
        next(map_blocks(len, [[1]], n_jobs=n_jobs))
