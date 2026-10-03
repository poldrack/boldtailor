"""The CLI and workflow package are documented where users look first."""

from pathlib import Path


def test_readme_and_user_guide_document_boldtailor_run():
    readme = Path("README.md").read_text()
    guide = Path("docs/user-guide.md").read_text()
    api = Path("docs/api.md").read_text()
    assert "boldtailor run --bids-dir" in readme and "boldtailor run --bids-dir" in guide
    assert "--modulator" in guide and "derivatives/boldtailor_hrf-" in guide
    assert "WorkflowSettings" in api and "run_workflow" in api and "render_report" in api
