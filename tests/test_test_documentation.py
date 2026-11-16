"""Keep test intent visible alongside executable checks."""

import ast
from pathlib import Path


def test_repository_tests_document_their_purpose():
    """Require a purpose docstring for each package, experiment and example test."""
    root = Path(__file__).resolve().parents[1]
    missing = []
    for directory in ("tests", "experiments", "examples"):
        for path in sorted((root / directory).rglob("test_*.py")):
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith("test_") and not ast.get_docstring(node):
                        missing.append(f"{path.relative_to(root)}::{node.name}")
    assert not missing, f"{len(missing)} tests need purpose docstrings: {missing[:10]}"
