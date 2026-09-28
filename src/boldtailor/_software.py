"""Software versions for provenance, resolved only when requested."""

from importlib import metadata


def package_version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        if name != "boldtailor":
            raise
        return "unknown"
