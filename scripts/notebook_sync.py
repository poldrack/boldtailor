"""Synchronize local HRF notebooks safely and require text-only, explicitly staged commits."""

import argparse
from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import jupytext
import nbformat
from jupytext.combine import combine_inputs_with_outputs


def git(root, *args):
    result = subprocess.run(["git", *args], cwd=root, capture_output=True)
    return result.stdout if result.returncode == 0 else None


def signature(book):
    return [(c.cell_type, c.source.rstrip()) for c in book.cells]


def text_book(book):
    result = deepcopy(book)
    result.metadata = {k: v for k, v in result.metadata.items() if k == "kernelspec"}
    result.metadata["jupytext"] = dict(
        formats="ipynb,py:percent",
        notebook_metadata_filter="kernelspec,jupytext",
        cell_metadata_filter="-all",
    )
    for cell in result.cells:
        cell.metadata = {}
    return result


def synchronize(root, notebook_path):
    script = notebook_path.with_suffix(".py")
    notebook = nbformat.read(notebook_path, as_version=4)
    if not script.exists():
        jupytext.write(text_book(notebook), script, fmt="py:percent")
        return script
    python = jupytext.read(script, fmt="py:percent")
    if signature(python) == signature(notebook):
        return script
    relative = script.relative_to(root).as_posix()
    original = git(root, "show", f"HEAD:{relative}")
    base = (
        signature(jupytext.reads(original.decode(), fmt="py:percent"))
        if original
        else None
    )
    if base == signature(python):
        jupytext.write(text_book(notebook), script, fmt="py:percent")
    elif base == signature(notebook):
        updated = combine_inputs_with_outputs(python, notebook, fmt="py:percent")
        nbformat.write(updated, notebook_path)
    else:
        raise ValueError(
            f"Input conflict in {relative}: both versions changed; resolve explicitly before committing."
        )
    return script


def check(root, check_tracked=False):
    staged = git(root, "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    if staged is None:
        raise ValueError("Cannot inspect the Git index.")
    if check_tracked:
        staged += git(root, "ls-files", "-z", "*.ipynb") or b""
    notebooks = [p.decode() for p in staged.split(b"\0") if p.endswith(b".ipynb")]
    if notebooks:
        raise ValueError(
            "Rendered .ipynb files must not be committed: " + ", ".join(notebooks)
        )
    pending = []
    paths = [
        path
        for directory in ("hrf_estimation", "hrf_simulations")
        for path in (root / "experiments" / directory).glob("*.ipynb")
    ]
    for path in sorted(paths):
        script = synchronize(root, path)
        relative = script.relative_to(root).as_posix()
        if git(root, "show", ":" + relative) != script.read_bytes():
            pending.append(relative)
    if pending:
        print(
            "Notebook sources synchronized. Review and git add these .py files, then retry:"
        )
        print("\n".join(pending))
        return 1
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--check-tracked",
        action="store_true",
        help="Also reject already tracked notebooks (CI).",
    )
    args = parser.parse_args()
    try:
        return check(args.root.resolve(), args.check_tracked)
    except (ValueError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
