"""Static checks that keep the NSD experiments on core boldtailor and nsdpreproc."""

import ast

FORBIDDEN_MODULES = ("boldtailor.workflow", "boldtailor.cli")
FORBIDDEN_FRAGMENTS = (
    "den-91k_bold",  # fMRIPrep CIFTI BOLD: *_space-fsLR_den-91k_bold.dtseries.nii
    "derivatives/ppdata",
    "func1pt8mm",
)


def _forbidden(module):
    return any(module == m or module.startswith(m + ".") for m in FORBIDDEN_MODULES)


def _imported(node):
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if isinstance(node, ast.ImportFrom) and node.module:
        return [node.module] + [f"{node.module}.{a.name}" for a in node.names]
    return []


def _import_problems(tree):
    return [
        f"line {node.lineno}: imports {name}"
        for node in ast.walk(tree)
        for name in _imported(node)
        if _forbidden(name)
    ]


def _string_problems(tree):
    return [
        f"line {node.lineno}: names {fragment!r}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
        for fragment in FORBIDDEN_FRAGMENTS
        if fragment in node.value
    ]


def violations(source, *, check_strings=True):
    """Boundary violations in one module's source text.

    ``check_strings=False`` checks imports only, for test modules that build
    decoy fMRIPrep or ppdata paths to prove they are never read.
    """
    tree = ast.parse(source)
    strings = _string_problems(tree) if check_strings else []
    return _import_problems(tree) + strings
