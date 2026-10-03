"""Workflow fixtures: synthetic BIDS datasets and a settings builder."""

import pytest

from boldtailor.hrf_library import HrfLibrary
from tests.workflow import synthetic_bids


@pytest.fixture
def confounds():
    return synthetic_bids.make_confounds()


@pytest.fixture
def events():
    return synthetic_bids.make_events()


@pytest.fixture
def dataset(tmp_path, confounds, events):
    return synthetic_bids.write_dataset(tmp_path, confounds, events)


@pytest.fixture
def four_runs(dataset):
    root, prep, *_ = dataset
    return synthetic_bids.add_runs_three_and_four(root, prep)


@pytest.fixture
def six_run_dataset(dataset):
    root, prep, *_ = dataset
    return synthetic_bids.make_six_runs(root, prep)


@pytest.fixture
def small_library():
    return HrfLibrary.from_parameters([[3, 10, 0.5, 0.5, 2, 0, 36]])


@pytest.fixture
def cv_library():
    return HrfLibrary.from_parameters([[4, 12, 0.8, 1, 5, 0, 36]])


@pytest.fixture
def settings_for():
    from boldtailor.workflow.settings import WorkflowSettings

    def build(root, **overrides):
        options = dict(
            subject="sub-07",
            session="ses-nsd10",
            task="nsdcore",
            n_jobs=1,
            block_size=2,
            surface_maps=False,
        )
        options.update(overrides)
        return WorkflowSettings(bids_dir=root, **options)

    return build
