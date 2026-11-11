"""Portable extraction must reproduce direct BIDS loading without source files."""

import shutil
import json
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from experiments.hrf_estimation.test_inputs import bids as bids  # noqa: F401


def test_roundtrip_all_subjects_sessions_and_offline_loading(bids, tmp_path):
    from experiments.hrf_simulations.bundle import export_bundle, inventory, load_session
    from experiments.hrf_estimation.inputs import discover_runs, load_run

    for relative in ('sub-07', 'derivatives/nsdpreproc/sub-07'):
        source = bids / relative / 'ses-nsd11'
        target = bids / relative.replace('sub-07', 'sub-01') / 'ses-nsd20'
        shutil.copytree(source, target)
        for path in target.rglob('*sub-07*'):
            path.rename(path.with_name(path.name.replace('sub-07', 'sub-01').replace('nsd11', 'nsd20')))
    direct = [load_run(s, np.array([2, 0])) for s in discover_runs(bids, 'sub-07', [11])]
    bundle = tmp_path / 'portable.h5'
    export_bundle(bids, bundle, np.array([2, 0]))
    assert inventory(bundle) == {'sub-01': [20], 'sub-07': [11]}
    shutil.rmtree(bids / 'derivatives')
    shutil.rmtree(bids / 'sub-07')
    loaded = load_session(bundle, 'sub-07', 11)
    for actual, expected in zip(loaded, direct, strict=True):
        np.testing.assert_array_equal(actual.signals, expected.signals)
        np.testing.assert_array_equal(actual.frame_times, expected.frame_times)
        np.testing.assert_array_equal(actual.feature_ids, expected.feature_ids)
        np.testing.assert_array_equal(actual.brain_axis.vertex, expected.brain_axis.vertex)
        pd.testing.assert_frame_equal(actual.events, expected.events)
        pd.testing.assert_frame_equal(actual.confounds, expected.confounds)
    one = load_session(bundle, 'sub-07', 11, first_vertex=True)
    assert one[0].signals.shape[1] == 1
    assert one[0].feature_ids.tolist() == [2]
    with pytest.raises(FileExistsError):
        export_bundle(bids, bundle, np.array([2, 0]))
    with pytest.raises(ValueError, match='session'):
        load_session(bundle, 'sub-07', 12)


def test_failed_export_is_not_published_and_skip_is_explicit(bids, tmp_path):
    from experiments.hrf_simulations.bundle import export_bundle, load_session
    next(bids.rglob('*run-01_events.tsv')).unlink()
    destination = tmp_path / 'partial.h5'
    with pytest.raises(FileNotFoundError):
        export_bundle(bids, destination, np.array([0]))
    assert not destination.exists()
    export_bundle(bids, destination, np.array([0]), skip_incomplete=True)
    assert len(load_session(destination, 'sub-07', 11)) == 1


def test_cli_bundle_path_and_makefile(tmp_path):
    from experiments.hrf_simulations.cli import _parser
    from experiments.hrf_simulations.test_makefile import dry_run
    args = _parser().parse_args(['run', '--bundle', str(tmp_path/'data.h5'), '--calibration', 'c', '--output', 'o'])
    assert args.bundle == tmp_path/'data.h5'
    assert '--bundle' in dry_run('workflow', 'BUNDLE=/tmp/data.h5', f'CALIBRATION={tmp_path / "new"}')


def test_reject_donor_calibration_from_other_subject(tmp_path):
    from experiments.hrf_simulations.cli import _check_calibration_subject
    (tmp_path / 'provenance.json').write_text(json.dumps({'subject': 'sub-07'}))
    with pytest.raises(ValueError, match='subject'):
        _check_calibration_subject(Namespace(calibration=tmp_path, subject='sub-01'))
    _check_calibration_subject(Namespace(calibration=tmp_path, subject='sub-07'))


def test_bundle_cli_loader_without_source_or_mask(bids, tmp_path):
    from experiments.hrf_simulations.bundle import export_bundle
    from experiments.hrf_simulations.cli import _session_runs, _inventory
    path = tmp_path / 'offline.h5'
    export_bundle(bids, path, np.array([0]))
    args = Namespace(bundle=path, subject='sub-07', sessions=[11], mask=tmp_path/'absent')
    shutil.rmtree(bids / 'derivatives')
    assert len(_session_runs(args, 11)) == 2
    assert len(_inventory(args)['bundle_sha256']) == 64
