import numpy as np
import pandas as pd
import pytest

from experiments.nsd_replication.trials import (
    images_with,
    out_of_sample_images,
    presentation_order,
    repeat_counts,
    repetition_array,
    trial_table,
)


def _events(onsets, images):
    return pd.DataFrame(
        {"onset": onsets, "duration": 3.0, "73k_id": images, "trial_type": 0}
    )


def test_trial_table_orders_runs_and_onsets():
    events = [_events([8.0, 4.0], [5, 6]), _events([4.0], [5])]
    table = trial_table(events, ["run-01", "run-02"], "ses-nsd10", "73k_id")
    assert list(table.columns) == ["session", "run", "trial", "onset", "image"]
    assert table["image"].tolist() == [6, 5, 5]
    assert table["trial"].tolist() == [0, 1, 0]
    assert table["run"].tolist() == ["run-01", "run-01", "run-02"]


def test_trial_table_missing_image_column_raises():
    with pytest.raises(ValueError, match="73k_id"):
        trial_table([_events([4.0], [1]).drop(columns="73k_id")], ["run-01"], "s", "73k_id")


def _trials(rows):
    return pd.DataFrame(rows, columns=["session", "run", "trial", "onset", "image"])


def test_repetition_array_uses_first_three_in_time_order():
    trials = _trials(
        [
            ("ses-b", "run-01", 0, 4.0, 9),  # 3rd presentation (later session)
            ("ses-a", "run-01", 0, 4.0, 9),  # 1st
            ("ses-a", "run-01", 1, 8.0, 9),  # 2nd (same run as the 1st)
            ("ses-c", "run-01", 0, 4.0, 9),  # 4th: ignored
        ]
    )
    betas = np.array([[3.0], [1.0], [2.0], [4.0]])
    assert presentation_order(trials).tolist() == [2, 0, 1, 3]
    reps = repetition_array(betas, trials, np.array([9]))
    assert reps.shape == (3, 1, 1)
    assert reps[:, 0, 0].tolist() == [1.0, 2.0, 3.0]


def test_images_with_and_counts():
    trials = _trials(
        [("s", "r", i, float(i), img) for i, img in enumerate([1, 1, 1, 2, 2, 3])]
    )
    assert images_with(trials, 3).tolist() == [1]
    assert images_with(trials, 2).tolist() == [1, 2]
    assert repeat_counts(trials) == {1: 1, 2: 1, 3: 1}


def test_out_of_sample_requires_distinct_sessions():
    trials = _trials(
        [
            ("s1", "r", 0, 0.0, 1), ("s2", "r", 0, 0.0, 1), ("s3", "r", 0, 0.0, 1),
            ("s1", "r", 1, 4.0, 2), ("s1", "r", 2, 8.0, 2), ("s2", "r", 1, 4.0, 2),
        ]
    )
    assert out_of_sample_images(trials).tolist() == [1]
