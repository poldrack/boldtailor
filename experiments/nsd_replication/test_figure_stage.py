import pandas as pd
import pytest

from experiments.nsd_replication import figure_stage
from experiments.nsd_replication.config import ExperimentConfig, select_sessions


@pytest.fixture
def config(tmp_path):
    return ExperimentConfig(
        bids_dir=tmp_path,
        output_dir=tmp_path / "out",
        freesurfer_dir=tmp_path,
        subjects=("sub-07",),
        sessions=("ses-a", "ses-b", "ses-c"),
    )


@pytest.fixture
def saved(monkeypatch):
    """Figure name -> the texts placed on the figure itself (not its axes)."""
    texts = {}

    def save(fig, path):
        texts[path.stem] = [t.get_text() for t in fig.texts]

    monkeypatch.setattr(figure_stage, "_save", save)
    return texts


def _subject_dir(config):
    return config.output_dir / "metrics" / "ppdata" / "sub-07"


def _write(path, table):
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, sep="\t", index=False)


def _write_r1(config, window):
    curves = pd.DataFrame(
        {
            "version": ["b1", "b4"],
            "threshold": [0.0, 0.0],
            "n_features": 9,
            "mean_difference": [0.0, 0.1],
        }
    )
    if window is not None:
        curves = curves.assign(sessions="|".join(window))
    _write(_subject_dir(config) / "r1.tsv", curves)


def _write_gate(config, sessions):
    gate = pd.DataFrame(
        {
            "condition": ["real", "null"] * len(sessions),
            "gate": "on",
            "n_components": range(2 * len(sessions)),
            "session": [s for s in sessions for _ in range(2)],
        }
    )
    _write(_subject_dir(config) / "features_gate.tsv", gate)


def test_footer_names_the_sessions_drawn(config, saved):
    _write_r1(config, config.sessions)
    figure_stage.make_figures(config)
    footer = "sessions: ses-a, ses-b, ses-c (3 of 3 in config)"
    assert saved["r1"] == [footer]


def test_footer_names_a_selected_subset(config, saved):
    selected = select_sessions(config, ["ses-c", "ses-a"])
    _write_r1(selected, selected.sessions)
    figure_stage.make_figures(selected)
    assert saved["r1"] == ["sessions: ses-a, ses-c (2 of 3 in config)"]


def test_metric_table_from_another_window_raises(config, saved):
    _write_r1(config, config.sessions)
    selected = select_sessions(config, ["ses-b"])
    with pytest.raises(ValueError, match=r"r1\.tsv.*ses-a\|ses-b\|ses-c.*--sessions"):
        figure_stage.make_figures(selected)
    assert saved == {}


def test_metric_table_without_a_window_raises(config, saved):
    _write_r1(config, None)
    with pytest.raises(ValueError, match=r"r1\.tsv.*no session"):
        figure_stage.make_figures(config)


def test_group_table_from_another_window_raises(config, saved):
    rsa = pd.DataFrame(
        {"threshold": [0.0], "subject_a": ["a"], "subject_b": ["b"], "r": [0.3]}
    )
    path = config.output_dir / "metrics" / "ppdata" / "group_sub-07" / "rsa_b4.tsv"
    _write(path, rsa.assign(sessions="ses-a"))
    with pytest.raises(ValueError, match=r"rsa_b4\.tsv.*ses-a"):
        figure_stage.make_figures(config)


def test_feature_rows_limited_to_selected_sessions(config, saved, monkeypatch):
    drawn = {}

    def fig_gate(table):
        drawn["sessions"] = set(table["session"])
        return figure_stage.plt.figure()

    monkeypatch.setattr(figure_stage.figures, "fig_gate", fig_gate)
    _write_gate(config, config.sessions)
    figure_stage.make_figures(select_sessions(config, ["ses-b"]))
    assert drawn["sessions"] == {"ses-b"}
    assert saved["gate"] == ["sessions: ses-b (1 of 3 in config)"]


def test_selected_session_missing_from_features_raises(config, saved):
    _write_gate(config, ("ses-a", "ses-b"))
    with pytest.raises(ValueError, match=r"features_gate\.tsv.*ses-c.*features"):
        figure_stage.make_figures(config)
