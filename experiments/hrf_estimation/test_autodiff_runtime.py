"""Autodiff runtime settings must reach actual worker processes."""
import importlib
import os

import pytest
from joblib import Parallel, delayed, parallel_config


def runtime_probe():
    import jax
    from joblib.externals.loky import process_executor
    return (os.getpid(), process_executor._MAX_MEMORY_LEAK_SIZE,
            jax.config.jax_compilation_cache_dir,
            jax.config.jax_persistent_cache_min_compile_time_secs)


def test_worker_receives_memory_guard_and_disk_cache(tmp_path):
    pytest.importorskip("jax")
    api = importlib.import_module("experiments.hrf_estimation.autodiff_runtime")
    limit = 8 * 1024**3
    cache = str(tmp_path / "compilation")
    with parallel_config(backend="loky", inner_max_num_threads=1):
        actual = Parallel(n_jobs=2, initializer=api.configure_worker,
                          initargs=(limit, cache))(delayed(runtime_probe)() for _ in range(4))
    assert all(pid != os.getpid() for pid, *_ in actual)
    assert all((guard, directory, threshold) == (limit, cache, 0)
               for _, guard, directory, threshold in actual)


def test_runtime_rejects_invalid_memory_limit(tmp_path):
    pytest.importorskip("jax")
    api = importlib.import_module("experiments.hrf_estimation.autodiff_runtime")
    with pytest.raises(ValueError, match="positive"):
        api.configure_worker(-1, str(tmp_path))
