"""The CLI and workflow package are documented where users look first."""

from pathlib import Path


def test_readme_and_user_guide_document_boldtailor_run():
    readme = Path("README.md").read_text()
    guide = Path("docs/user-guide.md").read_text()
    api = Path("docs/api.md").read_text()
    assert (
        "boldtailor run --bids-dir" in readme and "boldtailor run --bids-dir" in guide
    )
    assert "--modulator" in guide and "derivatives/boldtailor_hrf-" in guide
    assert (
        "WorkflowSettings" in api and "run_workflow" in api and "render_report" in api
    )


def test_user_guide_documents_categorical_modulators():
    guide = Path("docs/user-guide.md").read_text()
    api = Path("docs/api.md").read_text()
    assert "categorical" in guide and "reference=" in guide and "trial_type[" in guide
    assert 'kind="categorical"' in api and "level_name" in api


REPO = Path(__file__).resolve().parents[1]
LIVING_DOCS = [
    "README.md",
    "docs/README.md",
    "docs/user-guide.md",
    "docs/api.md",
    "docs/development.md",
    ".github/workflows/tests.yml",
]


def test_living_docs_point_at_bids_workflow_examples_and_two_kinds_of_code():
    for name in LIVING_DOCS:
        text = (REPO / name).read_text()
        assert "examples/NSD" not in text, name
        assert "examples.NSD" not in text, name
    assert (REPO / "examples/bids_workflow/README.md").is_file()
    assert not (REPO / "examples/NSD").exists()
    for name in ("README.md", "docs/development.md"):
        text = (REPO / name).read_text()
        assert "Two kinds of code" in text, name
        assert "experiments/nsd_replication" in text and "nsdpreproc" in text, name


def test_living_docs_describe_the_cli_as_bids_not_nsd_style():
    for name in LIVING_DOCS:
        assert "NSD-style" not in (REPO / name).read_text(), name


def test_api_reference_places_task_detection_names_in_core():
    rows = [
        line
        for line in (REPO / "docs/api.md").read_text().splitlines()
        if line.startswith("| `")
    ]
    for name in ("detect_task_model(", "task_model_notes(", "observed_levels("):
        (row,) = [r for r in rows if r.startswith(f"| `{name}")]
        assert "`boldtailor.task_detection`" in row, row
