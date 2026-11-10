"""Execute the report notebook with the active uv Python, without kernel install."""

import argparse
import os
from pathlib import Path
import sys

import jupytext
from jupyter_client import KernelManager
from nbclient import NotebookClient


def render_notebook(source, output, report):
    book = jupytext.read(source, fmt="py:percent")
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable,
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}",
    ]
    previous = os.environ.get("HRF_SIM_REPORT")
    os.environ["HRF_SIM_REPORT"] = str(Path(report).resolve())
    try:
        NotebookClient(book, km=manager, timeout=180).execute()
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        jupytext.write(book, output, fmt="ipynb")
    finally:
        if previous is None:
            os.environ.pop("HRF_SIM_REPORT", None)
        else:
            os.environ["HRF_SIM_REPORT"] = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).with_name("results.py")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    render_notebook(args.source, args.output, args.report)


if __name__ == "__main__":
    main()
