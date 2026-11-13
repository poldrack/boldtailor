"""Run one interleaved shard (cases i, i+n, i+2n, ...) of `cli run`.

Lives outside the fingerprinted modules (runner.implementation_fingerprint
globs only experiments/hrf_simulations/*.py), so shard outputs are identical
to, and resumable by, an unsharded run into the same output directory.

    python experiments/hrf_simulations/slurm/shard.py <index> <count> run ...
"""

import sys

from experiments.hrf_simulations import cli


def main(argv):
    index, count = int(argv[0]), int(argv[1])
    if not 0 <= index < count:
        raise ValueError("shard index must be in [0, count)")
    grid = cli.simulation_cases
    cli.simulation_cases = lambda *a, **k: grid(*a, **k)[index::count]
    cli.main(argv[2:])


if __name__ == "__main__":
    main(sys.argv[1:])
