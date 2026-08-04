"""Filesystem anchors, so scripts work from any working directory."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARCHIVE_V1 = ROOT / "archive_v1"


def result(name: str) -> str:
    """Absolute path to a result file, creating the directory if needed."""
    RESULTS.mkdir(exist_ok=True)
    return str(RESULTS / name)
