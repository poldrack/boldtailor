"""Real process execution must preserve scientific results and bounded work."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from examples.NSD.nsd_single_trial import run_single_trial_analysis
from examples.NSD.test_nsd_cifti import confounds, dataset, events  # noqa: F401
from examples.NSD.test_nsd_hrf_selection import hrf_nsd  # noqa: F401
from examples.NSD.test_nsd_single_trial import mini_nsd  # noqa: F401


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
    from examples.NSD.parallel_blocks import map_blocks

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


def _assert_same_artifacts(serial, parallel):
    other = {p.name: p for p in parallel}
    assert {p.name for p in serial} == set(other)
    for path in serial:
        if path.name.endswith(".dscalar.nii"):
            a, b = nib.load(path), nib.load(other[path.name])
            assert a.header.get_axis(0) == b.header.get_axis(0)
            assert a.header.get_axis(1) == b.header.get_axis(1)
            np.testing.assert_allclose(
                a.get_fdata(), b.get_fdata(), rtol=1e-7, atol=1e-7, equal_nan=True
            )
        elif path.suffix == ".tsv":
            pd.testing.assert_frame_equal(
                pd.read_csv(path, sep="\t"), pd.read_csv(other[path.name], sep="\t")
            )
        elif path.suffix == ".npz":
            with (
                np.load(path, allow_pickle=False) as a,
                np.load(other[path.name], allow_pickle=False) as b,
            ):
                assert a.files == b.files
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])


def test_parallel_canonical_cli_matches_serial(mini_nsd, tmp_path):
    root, prep, *_ = mini_nsd
    serial = run_single_trial_analysis(
        root, prep, tmp_path / "serial", ridge_alpha=0.1, block_size=1, n_jobs=1
    )
    output = tmp_path / "parallel"
    script = Path(__file__).with_name("nsd_single_trial.py")
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--bids-root",
            str(root),
            "--fmriprep-root",
            str(prep),
            "--output-root",
            str(output),
            "--ridge-alpha",
            "0.1",
            "--block-size",
            "1",
            "--n-jobs",
            "2",
        ],
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # The API lists published artifacts; the publisher's persistent lock is
    # private bookkeeping and is absent from that list in both modes.
    _assert_same_artifacts(
        serial,
        [
            p
            for p in output.rglob("*")
            if p.is_file() and ".boldtailor" not in p.relative_to(output).parts
        ],
    )
    metadata = json.loads(
        next(output.rglob("*desc-singletrialOLS_metadata.json")).read_text()
    )
    assert metadata["Execution"]["n_jobs"] == 2


def test_parallel_expanded_matches_serial_artifacts(hrf_nsd, tmp_path):
    root, prep, *_ = hrf_nsd
    outputs = [
        run_single_trial_analysis(
            root,
            prep,
            tmp_path / f"jobs-{jobs}",
            ridge_alpha=0.1,
            block_size=1,
            hrf_library="expanded",
            n_jobs=jobs,
        )
        for jobs in (1, 2)
    ]
    _assert_same_artifacts(*outputs)
    metadata = json.loads(
        next(
            p for p in outputs[1] if p.name.endswith("desc-hrfSelection_metadata.json")
        ).read_text()
    )
    assert metadata["Execution"]["n_jobs"] == 2
    assert metadata["Execution"]["block_size"] == 1


def test_parallel_worker_failure_publishes_nothing(hrf_nsd, tmp_path):
    root, prep, *_ = hrf_nsd
    path = next((root / "sub-07").rglob("*run-01_events.tsv"))
    table = pd.read_csv(path, sep="\t")
    table.loc[1, "onset"] = table.loc[0, "onset"]
    table.to_csv(path, sep="\t", index=False)
    output = tmp_path / "failed"
    output.mkdir()
    old = output / "old.txt"
    old.write_text("preserve")
    with pytest.raises(ValueError, match="eligible|rank"):
        run_single_trial_analysis(
            root,
            prep,
            output,
            ridge_alpha=0.1,
            block_size=1,
            hrf_library="expanded",
            n_jobs=2,
        )
    assert list(output.iterdir()) == [old]
    assert old.read_text() == "preserve"


@pytest.mark.parametrize("n_jobs", [0, -1, True, 1.5])
def test_invalid_worker_count_rejected_before_input_discovery(tmp_path, n_jobs):
    with pytest.raises(ValueError, match="n_jobs.*positive integer"):
        run_single_trial_analysis(tmp_path / "missing", n_jobs=n_jobs)
