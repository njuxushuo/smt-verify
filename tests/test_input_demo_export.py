from __future__ import annotations

import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_input_demos import (
    InputDemoLayoutError,
    discover_stage_inputs,
    export_input_demos,
)


INPUT_ROOT = ROOT / "input"
PROVED_FIXTURE = (
    INPUT_ROOT
    / "matmul_shard_to_partial"
    / "matmul_shard_to_partial.json"
)
REJECTED_FIXTURE = (
    INPUT_ROOT
    / "broadcast"
    / "matmul_batched"
    / "matmul_invalid_batch_broadcast_rejected"
    / "matmul_invalid_batch_broadcast_rejected.json"
)


def test_every_repository_json_has_a_sibling_demo() -> None:
    inputs = discover_stage_inputs(INPUT_ROOT)

    assert len(inputs) == 28
    assert all(path.parent.name == path.stem for path in inputs)
    reports = [path.parent / "demo.txt" for path in inputs]
    assert all(report.is_file() for report in reports)

    statuses = Counter()
    for input_path, report_path in zip(inputs, reports):
        report = report_path.read_text(encoding="utf-8")
        assert f"Input JSON: {input_path.name}" in report
        match = re.search(r"^  Verification: (\w+)$", report, re.MULTILINE)
        assert match is not None
        statuses[match.group(1)] += 1
    assert statuses == {"PROVED": 20, "DISPROVED": 6, "REJECTED": 2}


def test_batch_export_writes_complete_and_rejected_reports(tmp_path: Path) -> None:
    for source in (PROVED_FIXTURE, REJECTED_FIXTURE):
        target_directory = tmp_path / source.stem
        target_directory.mkdir()
        shutil.copy2(source, target_directory / source.name)

    outputs = export_input_demos(tmp_path)

    assert len(outputs) == 2
    reports = {output.parent.name: output.read_text(encoding="utf-8") for output in outputs}
    assert "Verification: PROVED" in reports["matmul_shard_to_partial"]
    assert "Verification: REJECTED" in reports["matmul_invalid_batch_broadcast_rejected"]


def test_batch_export_rejects_json_outside_its_stem_directory(tmp_path: Path) -> None:
    shutil.copy2(PROVED_FIXTURE, tmp_path / PROVED_FIXTURE.name)

    with pytest.raises(InputDemoLayoutError, match="expected parent directory"):
        export_input_demos(tmp_path)
