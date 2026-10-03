"""Software versions for provenance, resolved only when requested."""

from importlib import metadata


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        if name != "boldtailor":
            raise
        return "unknown"


def software_environment() -> dict[str, str]:
    """Interpreter, platform, and package versions, resolved at call time."""
    import platform

    packages = ("boldtailor", "numpy", "scipy", "pandas", "nilearn")
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        **{name: package_version(name) for name in packages},
    }
