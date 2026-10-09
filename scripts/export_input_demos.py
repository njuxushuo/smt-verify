"""Verify every input and generate sibling demo.txt reports."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.app.cli import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root", default=str(Path(__file__).resolve().parents[1] / "input")
    )
    parser.add_argument("--timeout-ms", default="30000")
    args = parser.parse_args()
    raise SystemExit(
        main(
            [
                "batch",
                args.input_root,
                "--write-reports",
                "--timeout-ms",
                args.timeout_ms,
            ]
        )
    )
