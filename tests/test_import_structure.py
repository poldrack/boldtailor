"""The package's import graph: acyclic, and honest about what is private."""

import ast
from graphlib import CycleError, TopologicalSorter
from pathlib import Path
from typing import NamedTuple

import pytest

import boldtailor

ROOT = Path(boldtailor.__file__).parent
LAZY_IMPORT_MODULES = {"boldtailor.cli"}  # keeps ``--help`` light


class Import(NamedTuple):
    importer: str
    target: str
    name: str
    line: int
    top_level: bool


def _module_name(path):
    parts = path.relative_to(ROOT.parent).with_suffix("").parts
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _target(importer, path, node):
    if not node.level:
        return node.module or ""
    package = importer.split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    base = package[: len(package) - node.level + 1]
    return ".".join(base + ([node.module] if node.module else []))


def _from_imports(importer, path, tree, modules):
    top = {id(node) for node in tree.body}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ImportFrom):
            continue
        target = _target(importer, path, node)
        for alias in node.names:
            submodule = f"{target}.{alias.name}"
            owner = submodule if submodule in modules else target
            name = "" if owner == submodule else alias.name
            yield Import(importer, owner, name, node.lineno, id(node) in top)


def _plain_imports(importer, tree):
    top = {id(node) for node in tree.body}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield Import(importer, alias.name, "", node.lineno, id(node) in top)


@pytest.fixture(scope="module")
def sources():
    paths = {_module_name(path): path for path in ROOT.rglob("*.py")}
    return {name: (path, ast.parse(path.read_text())) for name, path in paths.items()}


@pytest.fixture(scope="module")
def imports(sources):
    found = []
    for importer, (path, tree) in sources.items():
        found += _from_imports(importer, path, tree, sources)
        found += _plain_imports(importer, tree)
    return [item for item in found if item.target in sources]


def _defined_names(tree):
    names = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign | ast.AnnAssign):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(t.id for t in targets if isinstance(t, ast.Name))
    return names


def _declared_all(tree):
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets
        ):
            return set(ast.literal_eval(node.value))
    return set()


def test_package_import_graph_has_no_cycles(imports):
    """No module may depend on itself through any chain of imports.

    Function-level imports count: a lazy import only hides a cycle at runtime.
    """
    graph = {}
    for item in imports:
        graph.setdefault(item.importer, set()).add(item.target)
    try:
        tuple(TopologicalSorter(graph).static_order())
    except CycleError as error:
        pytest.fail(f"import cycle: {' -> '.join(error.args[1])}")


def test_no_module_imports_a_private_name_of_another_module(imports):
    """A leading underscore means module-private, so no other module imports it."""
    offenders = [
        f"{item.importer}:{item.line} imports {item.target}.{item.name}"
        for item in imports
        if item.name.startswith("_") and item.importer != item.target
    ]
    assert not offenders, "\n".join(sorted(offenders))


def test_package_imports_are_at_module_level_outside_the_cli(imports):
    """Only the CLI defers package imports; elsewhere every dependency is visible."""
    offenders = [
        f"{item.importer}:{item.line} lazily imports {item.target}"
        for item in imports
        if not item.top_level and item.importer not in LAZY_IMPORT_MODULES
    ]
    assert not offenders, "\n".join(sorted(offenders))


def test_workflow_modules_export_only_names_they_define(sources):
    """No workflow module is a facade: its ``__all__`` lists only its own names."""
    offenders = [
        f"{name} re-exports {sorted(_declared_all(tree) - _defined_names(tree))}"
        for name, (_, tree) in sources.items()
        if name.startswith("boldtailor.workflow.")
        and _declared_all(tree) - _defined_names(tree)
    ]
    assert not offenders, "\n".join(sorted(offenders))
