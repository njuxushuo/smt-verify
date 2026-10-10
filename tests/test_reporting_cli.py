import contextlib
import io
import json
import os
import re
import tempfile
import unittest
from pathlib import Path

import z3

from src.app.cli import main
from src.app.reporting import demo_path, render_report
from tests.support import INPUT, INPUT_CASE_COUNT, identity_stage
from tests.test_proofs import prove


class ReportingTests(unittest.TestCase):
    def test_report_contains_replayable_actual_solver_queries(self):
        result = prove(identity_stage())
        report = render_report(result)
        self.assertIn("Original tensor shapes", report)
        self.assertIn("Rin[0] SMT-LIB", report)
        self.assertIn("No coordinate is fixed to a sample", report)
        self.assertIn("Physical shape reduction: NONE", report)
        queries = re.findall(
            r"BEGIN SMT-LIB \(complete query\)\n(.*?)\nEND SMT-LIB", report, re.S
        )
        self.assertEqual(len(queries), 2)
        for query in queries:
            solver = z3.Solver()
            solver.from_string(query)
            self.assertEqual(solver.check(), z3.unsat)
        self.assertEqual(report, render_report(result))

    def test_counterexample_report_and_sat_replay(self):
        raw = identity_stage()
        raw["input_relations"] = []
        result = prove(raw)
        report = render_report(result)
        self.assertIn("Counterexample in original-shape", report)
        self.assertIn("DISPROVED", report)
        query = re.search(
            r"BEGIN SMT-LIB \(complete query\)\n(.*?)\nEND SMT-LIB", report, re.S
        ).group(1)
        solver = z3.Solver()
        solver.from_string(query)
        self.assertEqual(solver.check(), z3.sat)

    def test_unique_report_destinations_for_repository_inputs(self):
        paths = list(INPUT.rglob("*.json"))
        self.assertEqual(len(paths), INPUT_CASE_COUNT)
        self.assertEqual(len({demo_path(p) for p in paths}), INPUT_CASE_COUNT)
        for p in paths:
            self.assertEqual(demo_path(p).name, "demo.txt")

    def test_cli_batch_writes_multiple_same_directory_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("a", "b"):
                raw = identity_stage()
                raw["name"] = name
                (root / f"{name}.json").write_text(json.dumps(raw))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["batch", str(root), "--write-reports"]), 0)
            for name in ("a", "b"):
                text = (root / name / "demo.txt").read_text()
                self.assertIn(f"Stage: {name}", text)

    def test_cli_refuses_to_overwrite_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "stage.json"
            p.write_text(json.dumps(identity_stage()))
            before = p.read_bytes()
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                main(["verify", str(p), "--report", str(p)])
            self.assertEqual(p.read_bytes(), before)

    def test_cli_refuses_hardlink_report_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "stage.json"
            source.write_text(json.dumps(identity_stage()))
            alias = Path(tmp) / "alias.txt"
            os.link(source, alias)
            before = source.read_bytes()
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                main(["verify", str(source), "--report", str(alias)])
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(alias.read_bytes(), before)
