"""The HTML report is self-contained and covers every stage."""

import shlex
import warnings
from html.parser import HTMLParser

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from boldtailor.model import Modulator, TaskModel
from boldtailor.workflow import report

EMPTY = dict(
    runs=[],
    task_model=TaskModel(),
    library=None,
    glm_summary=None,
    hrf_summary=None,
    reliability=None,
    tuning=None,
    activation_summary=None,
    rt_summary=None,
    figures={},
    skipped=[],
    manifest=[],
)


class _Collector(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids, self.images = [], 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "section" and "id" in attrs:
            self.ids.append(attrs["id"])
        if tag == "img":
            assert attrs["src"].startswith("data:image/png;base64,")
            self.images += 1


def _render(settings, **overrides):
    return report.render_report(settings, **{**EMPTY, **overrides})


def _section_text(html, name):
    start = html.index(f'<section id="{name}">')
    return html[start : html.index("</section>", start)]


def test_report_has_a_section_per_stage_embedded_figures_and_a_manifest(
    bids_settings,
):
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    html = _render(
        bids_settings,
        glm_summary=pd.DataFrame({"model": ["CanonicalGLM"], "median": [0.1]}),
        figures={"Design": fig},
        skipped=[("reliability", "only one odd run")],
        manifest=[
            (
                "sub-07/ses-nsd10/func/a.dscalar.nii",
                "sub-07/ses-nsd10/func/a_provenance.json",
            )
        ],
    )
    plt.close(fig)
    parser = _Collector()
    parser.feed(html)
    assert parser.ids == [
        "settings",
        "inputs",
        "glms",
        "reliability",
        "betas",
        "summaries",
        "skipped",
        "files",
    ]
    assert parser.images == 1
    assert "only one odd run" in html and "a.dscalar.nii" in html
    assert "CanonicalGLM" in html
    assert "<link" not in html and "<script src" not in html


def test_embed_figure_does_not_warn(recwarn):
    fig, ax = plt.subplots()
    ax.plot([0, 1])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        tag = report.embed_figure(fig)
    plt.close(fig)
    assert tag.startswith('<img alt="figure" src="data:image/png;base64,')


def test_command_line_always_names_the_dataset_and_omits_defaults(bids_settings):
    words = shlex.split(report.command_line(bids_settings))
    assert words[:2] == ["boldtailor", "run"]
    assert words[words.index("--bids-dir") + 1] == str(bids_settings.bids_dir)
    assert words[words.index("--subject") + 1] == "sub-07"
    assert words[words.index("--session") + 1] == "ses-nsd10"
    assert words[words.index("--task") + 1] == "nsdcore"
    assert "--ridge-mode" not in words and "--hrf-library" not in words
    assert "--skip-stage" not in words


def test_command_line_reflects_non_default_settings(settings_for, dataset, tmp_path):
    root, *_ = dataset
    out = tmp_path / "out dir"
    mesh = tmp_path / "left.surf.gii"
    right = tmp_path / "right.surf.gii"
    settings = settings_for(
        root,
        output_dir=out,
        modulators=(Modulator("rt"), Modulator("score", missing="indicator")),
        hrf_library="sobol",
        hrf_n_samples=256,
        hrf_seed=3,
        hrf_selection_rt=False,
        ridge_mode="fixed",
        ridge_alpha=0.5,
        ridge_fractions=(0.5, 1.0),
        ridge_alphas=(0.1, 1.0),
        ridge_percentile=80.0,
        encoding_mode="absolute",
        stages=frozenset({"glms", "betas"}),
        surface_maps=False,
        surface_meshes={"left": mesh, "right": right},
        max_grayordinates=100,
        existing_results="overwrite",
    )
    words = shlex.split(report.command_line(settings))
    assert words[words.index("--output-dir") + 1] == str(out)
    assert words.count("--modulator") == 2
    assert "rt" in words and "score:indicator" in words
    assert words[words.index("--hrf-library") + 1] == "sobol"
    assert words[words.index("--hrf-n-samples") + 1] == "256"
    assert words[words.index("--hrf-seed") + 1] == "3"
    assert "--no-rt-in-hrf-selection" in words
    assert words[words.index("--ridge-mode") + 1] == "fixed"
    assert words[words.index("--ridge-alpha") + 1] == "0.5"
    assert words[words.index("--ridge-fractions") + 1 :][:2] == ["0.5", "1"]
    assert words[words.index("--ridge-alphas") + 1 :][:2] == ["0.1", "1"]
    assert words[words.index("--ridge-percentile") + 1] == "80"
    assert words[words.index("--encoding-mode") + 1] == "absolute"
    assert words[words.index("--skip-stage") + 1] == "reliability"
    assert words.count("--skip-stage") == 2
    assert "summaries" in words
    assert "--no-surface-maps" in words
    assert f"left={mesh}" in words and f"right={right}" in words
    assert words[words.index("--n-jobs") + 1] == "1"
    assert words[words.index("--block-size") + 1] == "2"
    assert words[words.index("--max-grayordinates") + 1] == "100"
    assert words[words.index("--existing-results") + 1] == "overwrite"


def test_command_line_quotes_paths_with_spaces(settings_for, dataset, tmp_path):
    root, *_ = dataset
    settings = settings_for(root, output_dir=tmp_path / "out dir")
    assert shlex.quote(str(tmp_path / "out dir")) in report.command_line(settings)


def test_settings_section_shows_the_command_line(bids_settings):
    html = _render(bids_settings)
    assert "boldtailor run" in _section_text(html, "settings")


def test_disabled_stage_section_is_a_one_line_note(settings_for, dataset):
    root, *_ = dataset
    settings = settings_for(root, stages=frozenset({"glms"}))
    html = _render(settings, tuning=pd.DataFrame({"alpha": [1.0]}))
    for name in ("reliability", "betas", "summaries"):
        assert "Stage disabled." in _section_text(html, name)
    assert "Stage disabled." not in _section_text(html, "glms")
    assert "<table" not in _section_text(html, "betas")


def test_skipped_stage_section_names_the_reason(bids_settings):
    html = _render(bids_settings, skipped=[("betas", "no usable runs")])
    section = _section_text(html, "betas")
    assert "Skipped: no usable runs." in section
    assert "<table" not in section
    assert "Stage disabled." not in html
