"""Generate one sibling demo.txt for every Stage JSON below input/."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.export_demo_report import export_demo_report


DEFAULT_INPUT_ROOT = PROJECT_ROOT / "input"


class InputDemoLayoutError(ValueError):
    """Raised when an input JSON does not have its required stem directory."""


def discover_stage_inputs(input_root: str | Path = DEFAULT_INPUT_ROOT) -> tuple[Path, ...]:
    """Return all Stage JSON inputs in deterministic path order."""

    root = Path(input_root)
    return tuple(sorted(root.rglob("*.json")))


def export_input_demos(input_root: str | Path = DEFAULT_INPUT_ROOT) -> tuple[Path, ...]:
    """Write one complete or rejected pipeline report beside every input JSON."""

    input_paths = discover_stage_inputs(input_root)
    for input_path in input_paths:
        if input_path.parent.name != input_path.stem:
            raise InputDemoLayoutError(
                f"{input_path}: expected parent directory named {input_path.stem!r}"
            )

    outputs = []
    for input_path in input_paths:
        outputs.append(
            export_demo_report(input_path.parent / "demo.txt", (input_path,))
        )
    return tuple(outputs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        default=DEFAULT_INPUT_ROOT,
        help="root containing one stem directory per Stage JSON",
    )
    arguments = parser.parse_args(argv)
    outputs = export_input_demos(arguments.input_root)
    print(f"Generated demos: {len(outputs)}")
    for output in outputs:
        print(f"  {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
