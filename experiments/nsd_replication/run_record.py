"""One JSON record per command run: what was asked for, including the sessions."""

from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import subprocess


def git_commit():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent,
        )
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _record(config, args, levels, started):
    return dict(
        command=args.command,
        source=args.source,
        config=str(Path(args.config).resolve()),
        levels=list(levels),
        subjects=list(config.subjects),
        sessions_run=list(config.sessions),
        config_sessions=list(config.configured_sessions),
        session_subset=config.sessions != config.configured_sessions,
        started_at=started.isoformat(),
        boldtailor_version=version("boldtailor"),
        git_commit=git_commit(),
    )


def _write(path, record):
    path.write_text(json.dumps(record, indent=2))


def start_run_record(config, args, levels, now=None):
    """Write ``run_records/<UTC start>_<command>_<source>.json``; return its path."""
    started = now or datetime.now(timezone.utc)
    stamp = started.strftime("%Y%m%dT%H%M%S%fZ")
    path = (
        config.output_dir / "run_records" / f"{stamp}_{args.command}_{args.source}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    _write(path, _record(config, args, levels, started))
    return path


def complete_run_record(path):
    """Mark the run as finished; a run that failed keeps no ``completed_at``."""
    record = json.loads(path.read_text())
    record["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write(path, record)
