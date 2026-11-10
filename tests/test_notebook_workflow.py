"""Git safeguards must synchronize notebook inputs without staging or data loss."""

from pathlib import Path
import subprocess
import sys
import ast

import jupytext
import nbformat
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/notebook_sync.py"


def git(root, *args):
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init")
    git(tmp_path, "config", "user.email", "test@example.com")
    git(tmp_path, "config", "user.name", "Notebook test")
    (tmp_path / ".gitignore").write_text("*.ipynb\n")
    folder = tmp_path / "experiments/hrf_estimation"
    folder.mkdir(parents=True)
    book = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell("x = 1")])
    book.metadata["comparison_records"] = [{"large": "do not export"}]
    book.cells[0].outputs = [
        nbformat.v4.new_output("stream", name="stdout", text="keep locally\n")
    ]
    nbformat.write(book, folder / "demo.ipynb")
    git(tmp_path, "add", ".gitignore")
    git(tmp_path, "commit", "-m", "initial")
    return tmp_path


def sync(repo):
    assert SCRIPT.exists(), "notebook synchronization safeguard is missing"
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(repo)],
        capture_output=True,
        text=True,
    )


def commit_pair(repo):
    assert sync(repo).returncode == 1  # Conversion asks for explicit staging.
    git(repo, "add", "experiments/hrf_estimation/demo.py")
    git(repo, "commit", "-m", "text source")


def test_ignored_notebooks_export_only_inputs_and_require_staging(repo):
    result = sync(repo)
    assert result.returncode == 1
    path = repo / "experiments/hrf_estimation/demo.py"
    source = path.read_text()
    ast.parse(source)
    assert "x = 1" in source
    assert "keep locally" not in source and "comparison_records" not in source
    assert git(repo, "diff", "--cached", "--name-only") == ""
    git(repo, "add", str(path.relative_to(repo)))
    assert sync(repo).returncode == 0
    book = nbformat.read(path.with_suffix(".ipynb"), as_version=4)
    assert book.cells[0].outputs[0].text == "keep locally\n"


def test_hook_rejects_staged_notebook_even_if_forced(repo):
    git(repo, "add", "-f", "experiments/hrf_estimation/demo.ipynb")
    result = sync(repo)
    assert result.returncode != 0
    assert "ipynb" in result.stdout + result.stderr


def test_divergent_edits_are_not_silently_overwritten(repo):
    commit_pair(repo)
    script = repo / "experiments/hrf_estimation/demo.py"
    script.write_text(script.read_text().replace("x = 1", "x = 2"))
    book = nbformat.read(script.with_suffix(".ipynb"), as_version=4)
    book.cells[0].source = "x = 3"
    nbformat.write(book, script.with_suffix(".ipynb"))
    before = [p.read_bytes() for p in [script, script.with_suffix(".ipynb")]]
    result = sync(repo)
    assert (
        result.returncode != 0 and "conflict" in (result.stdout + result.stderr).lower()
    )
    assert before == [p.read_bytes() for p in [script, script.with_suffix(".ipynb")]]


def test_notebook_edits_update_python_without_altering_staged_content(repo):
    commit_pair(repo)
    script = repo / "experiments/hrf_estimation/demo.py"
    original = git(repo, "show", "HEAD:experiments/hrf_estimation/demo.py")
    book = nbformat.read(script.with_suffix(".ipynb"), as_version=4)
    book.cells[0].source = "x = 3"
    nbformat.write(book, script.with_suffix(".ipynb"))
    assert sync(repo).returncode == 1
    assert "x = 3" in script.read_text()
    assert git(repo, "show", ":experiments/hrf_estimation/demo.py") == original


def test_python_edit_updates_local_notebook_and_keeps_unaffected_outputs(repo):
    commit_pair(repo)
    path = repo / "experiments/hrf_estimation/demo.py"
    path.write_text(path.read_text() + "\n# %%\ny = 2\n")
    git(repo, "add", str(path.relative_to(repo)))
    assert sync(repo).returncode == 0
    book = nbformat.read(path.with_suffix(".ipynb"), as_version=4)
    assert book.cells[-1].source == "y = 2"
    assert book.cells[0].outputs[0].text == "keep locally\n"


def test_python_only_checkout_needs_no_rendered_notebooks(repo):
    commit_pair(repo)
    (repo / "experiments/hrf_estimation/demo.ipynb").unlink()
    assert sync(repo).returncode == 0


def test_simulation_notebooks_use_same_text_only_workflow(repo):
    commit_pair(repo)
    folder = repo/'experiments/hrf_simulations'
    folder.mkdir()
    book = nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell('simulation = 1')])
    nbformat.write(book,folder/'results.ipynb')
    assert sync(repo).returncode == 1
    assert 'simulation = 1' in (folder/'results.py').read_text()
    git(repo,'add','experiments/hrf_simulations/results.py')
    assert sync(repo).returncode == 0


def test_ci_rejects_committed_rendered_notebooks(repo):
    git(repo, "add", "-f", "experiments/hrf_estimation/demo.ipynb")
    git(repo, "commit", "-m", "accidental rendered notebook")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(repo), "--check-tracked"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "demo.ipynb" in result.stdout + result.stderr
