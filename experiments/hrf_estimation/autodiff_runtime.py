"""Explicit, process-local runtime configuration for autodiff experiments."""

from pathlib import Path


def configure_worker(memory_growth_limit_bytes, cache_dir):
    """Allow JAX cache growth while retaining loky's worker recycling guard.

    Loky exposes no public setting for this threshold. Keep the private override
    isolated here and validate its presence rather than silently ignoring it.
    """
    if not isinstance(memory_growth_limit_bytes, int) or memory_growth_limit_bytes <= 0:
        raise ValueError("worker memory growth limit must be a positive integer")
    import jax
    from joblib.externals.loky import process_executor

    if not hasattr(process_executor, "_MAX_MEMORY_LEAK_SIZE"):
        raise RuntimeError("installed loky does not expose the expected memory guard")
    directory = Path(cache_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    process_executor._MAX_MEMORY_LEAK_SIZE = memory_growth_limit_bytes
    jax.config.update("jax_compilation_cache_dir", str(directory))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
