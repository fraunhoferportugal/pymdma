import json
from pathlib import Path


def jsonl_files(file: Path, path_col: str = "image"):
    """Extracts the image paths from a jsonl dataset file."""
    with file.open("r") as f:
        return [Path(json.loads(line)[path_col]) for line in f]
