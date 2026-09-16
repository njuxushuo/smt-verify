from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_demo_report import build_demo_report, export_demo_report


INVALID_FIXTURE = (
    ROOT
    / "input"
    / "broadcast"
    / "element_wise"
    / "add_incompatible_shapes_rejected"
    / "add_incompatible_shapes_rejected.json"
)


def test_demo_report_contains_real_pipeline_sections_and_outcomes(tmp_path: Path) -> None:
    output = export_demo_report(tmp_path / "demo_report.txt")
    report = output.read_text(encoding="utf-8")

    assert output.is_file()
    assert "matmul_shard_to_partial" in report
    assert "matmul_shard_to_partial_large" in report
    assert "matmul_shard_wrong_replicate" in report
    assert "[3] Reduced Shapes" in report
    assert "Normalized shape constraints:" in report
    assert "single__A__d0 >= 1" in report
    assert "single__A__d1 == single__B__d0" in report
    assert "[4] Symbolic Execution" in report
    assert "[5] Relation Encoding" in report
    assert "[6] SMT Verification" in report
    assert "[8] Certified Lemma" in report
    assert "Verification: PROVED" in report
    assert "Verification: DISPROVED" in report
    assert "Failed output constraints:" in report
    assert "Summary" in report


def test_demo_report_records_static_validation_rejection() -> None:
    report = build_demo_report((INVALID_FIXTURE,))

    assert "Input JSON: add_incompatible_shapes_rejected.json" in report
    assert "Stage loading: REJECTED" in report
    assert "Failure phase: Stage loading and static validation" in report
    assert "not broadcast-compatible" in report
    assert "Remaining pipeline stages: NOT RUN" in report
    assert "Verification: REJECTED" in report
    assert "Lemma: NOT GENERATED" in report
