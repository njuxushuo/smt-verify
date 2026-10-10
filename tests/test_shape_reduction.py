import copy
import contextlib
import io
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import z3

from src.app.bootstrap import default_registry
from src.app.cli import main
from src.app.reporting import demo_path, render_report
from src.core.errors import InvalidInput, Unsupported
from src.core.tensor import TensorRef
from src.ops.registry import OpRegistry
from src.ops.view import View
from src.smt.blocks import BlockContext
from src.smt.elements import ElementContext
from src.smt.solver import SolverOutcome
from src.stage.codec import stage_from_dict, stage_to_dict
from src.verification.result import Status
from src.verification.verifier import verify_file, verify_stage
from tests.support import (
    INPUT,
    fixture,
    identity_binding,
    identity_stage,
    partial,
    phi,
    piece,
)


def reduced(raw, **kwargs):
    return verify_stage(
        stage_from_dict(raw), default_registry(), shape_reduce=True, **kwargs
    )


class ShapeReductionTests(unittest.TestCase):
    def test_parse_failure_retains_requested_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.json"
            for contents in (None, "{broken"):
                if contents is not None:
                    path.write_text(contents)
                result = verify_file(path, default_registry(), shape_reduce=True)
                self.assertEqual(result.status, Status.INVALID_INPUT)
                self.assertEqual(result.mode, "SHAPE_REDUCTION_NOT_STARTED")

    def test_all_repository_inputs_reduce_and_certify_without_fallback(self):
        for path in sorted(INPUT.rglob("*.json")):
            with self.subTest(case=path.stem):
                original = path.read_bytes()
                result = verify_file(path, default_registry(), shape_reduce=True)
                self.assertEqual(result.status, Status.PROVED, result.reason)
                self.assertEqual(result.mode, "SHAPE_REDUCED_ELEMENTWISE")
                attempt = result.shape_attempt
                self.assertIsNone(attempt.fallback_reason)
                self.assertLessEqual(
                    attempt.plan.reduced_elements, attempt.plan.original_elements
                )
                self.assertEqual(attempt.reduced.scope, "REDUCED_SHAPES_ONLY")
                self.assertTrue(all(q.status == "UNSAT" for _, q in result.queries))
                self.assertEqual(path.read_bytes(), original)

    def test_fc2_joint_dimensions_and_contraction_are_reduced(self):
        raw = fixture("mlp_dp2_tp4_03_fc2_partial")
        original = copy.deepcopy(raw)
        result = reduced(raw)
        plan = result.shape_attempt.plan
        self.assertEqual(plan.stage.single.tensors["Z"].shape, (4, 2, 8))
        self.assertEqual(plan.stage.distributed[0].tensors["Z0"].shape, (2, 2, 2))
        self.assertEqual(plan.stage.single.tensors["W2"].shape, (8, 2))
        self.assertEqual(plan.block_size, 128)
        self.assertTrue(result.shape_attempt.block_definitions)
        self.assertEqual(raw, original)
        self.assertEqual(stage_to_dict(result.validated.stage), original)

    def test_cp_sp_chain_is_preserved_at_all_levels(self):
        result = reduced(fixture("dp2_cp2_sp2_zigzag_allgather"))
        raw = stage_to_dict(result.shape_attempt.plan.stage)
        terms = raw["input_relations"][0]["relation"]["terms"]
        self.assertEqual([t["family"] for t in terms], ["dp", "sp", "cp"])
        self.assertEqual(terms[1]["expr"]["maps"][0]["single_extent"], 4)
        self.assertEqual(terms[2]["expr"]["maps"][0]["local_extent"], 4)
        self.assertEqual(terms[2]["expr"]["maps"][0]["pieces"][1], piece(2, 4, 6))
        self.assertEqual(raw["single"]["tensors"]["X"]["shape"], [4, 8, 2])

    def test_mapping_error_is_not_erased_by_rounding(self):
        raw = identity_stage((4,), world=1)
        raw["distributed"]["ranks"]["0"]["tensors"]["X0"]["shape"] = [2]
        raw["input_relations"][0]["relation"]["terms"] = [
            phi(0, 2, 4, [[piece(0, 2, 0)]])
        ]
        raw["output_relations"][0]["relation"]["terms"] = [
            phi(0, 2, 4, [[piece(0, 2, 1)]])
        ]
        result = reduced(raw)
        self.assertEqual(result.status, Status.DISPROVED)
        self.assertEqual(result.mode, "SYMBOLIC_INDEX_FALLBACK")
        self.assertEqual(
            result.shape_attempt.plan.stage.single.tensors["X"].shape, (4,)
        )
        self.assertIsNotNone(result.counterexample)

    def test_partial_contributions_are_not_replicas(self):
        raw = identity_stage((8,))
        raw["input_relations"][0]["relation"]["terms"] = [partial()]
        result = reduced(raw)
        self.assertEqual(result.status, Status.DISPROVED)
        self.assertEqual(result.shape_attempt.reduced.status, Status.DISPROVED)

    def test_wrong_cp_output_mapping_remains_disproved(self):
        raw = fixture("dp2_cp2_sp2_zigzag_allgather")
        mapping = raw["output_relations"][0]["relation"]["terms"][1]["expr"]["maps"][0]
        mapping["pieces"] = [piece(0, 4, 12), piece(4, 8, 0)]
        result = reduced(raw)
        self.assertEqual(result.status, Status.DISPROVED)
        self.assertEqual(result.shape_attempt.reduced.status, Status.DISPROVED)

    def test_nested_matmul_falls_back_instead_of_assuming_a_lifting_theorem(self):
        raw = identity_stage((8, 8), world=1)
        for rank, program in [
            (None, raw["single"]),
            *raw["distributed"]["ranks"].items(),
        ]:
            x = "X" if rank is None else "X0"
            program["tensors"].update({"H": {"shape": [8, 8]}, "Y": {"shape": [8, 8]}})
            program["ops"] = [
                {"type": "matmul", "inputs": [x, x], "outputs": ["H"], "attrs": {}},
                {"type": "matmul", "inputs": ["H", x], "outputs": ["Y"], "attrs": {}},
            ]
            program["outputs"] = ["Y"]
        raw["output_relations"][0].update(
            single_tensor="Y", distributed_tensors={"0": "Y"}
        )
        result = reduced(raw)
        self.assertEqual(result.status, Status.PROVED)
        self.assertEqual(result.mode, "SYMBOLIC_INDEX_FALLBACK")
        self.assertIn("nested", result.shape_attempt.fallback_reason)

    def test_partial_permutations_and_global_gaps_survive(self):
        raw = identity_stage((8,))
        for program in raw["distributed"]["ranks"].values():
            next(iter(program["tensors"].values()))["shape"] = [4]
        raw["topology"]["families"]["layout"] = {"groups": ["g"]}
        terms = [
            phi(
                0,
                4,
                8,
                [[piece(0, 2, 0), piece(2, 4, 4)], [piece(0, 2, 4), piece(2, 4, 0)]],
                family="layout",
            ),
            partial(),
        ]
        for role in ("input_relations", "output_relations"):
            raw[role][0]["relation"]["terms"] = copy.deepcopy(terms)
        result = reduced(raw)
        self.assertEqual(result.status, Status.PROVED)
        self.assertEqual(result.mode, "SHAPE_REDUCED_ELEMENTWISE")
        rel = result.shape_attempt.reduced.validated.inputs[0]
        self.assertEqual(rel.image(0), (((0, 1), (2, 3)),))
        self.assertEqual(rel.image(0), rel.image(1))
        self.assertNotEqual(rel.pieces(0, 0), rel.pieces(1, 0))

    def test_missing_input_assumption_still_fails(self):
        raw = identity_stage((8,))
        raw["input_relations"] = []
        self.assertEqual(reduced(raw).status, Status.DISPROVED)

    def test_budget_and_optimizer_unknown_fall_back_explicitly(self):
        raw = identity_stage((8,))
        result = reduced(raw, max_reduced_elements=1)
        self.assertEqual(result.status, Status.PROVED)
        self.assertEqual(result.mode, "SYMBOLIC_INDEX_FALLBACK")
        self.assertIn("budget", result.shape_attempt.fallback_reason)
        with patch(
            "src.smt.shapes.ShapeConstraints.solve",
            side_effect=Unsupported("optimizer unknown"),
        ):
            result = reduced(raw)
        self.assertEqual(result.mode, "SYMBOLIC_INDEX_FALLBACK")
        self.assertIn("unknown", result.shape_attempt.fallback_reason)

    def test_operation_budget_is_not_bypassed(self):
        result = reduced(fixture("mlp_dp2_tp4_03_fc2_partial"), max_symbolic_ops=1)
        self.assertEqual(result.status, Status.UNSUPPORTED)

    def test_original_certificate_checks_actual_block_count_budget(self):
        raw = identity_stage(world=1)
        shapes = {
            "A": [2, 4],
            "B": [4, 2],
            "C": [2, 6],
            "D": [6, 2],
            "P": [2, 2],
            "Q": [2, 2],
        }
        for suffix, program in [
            ("", raw["single"]),
            ("0", raw["distributed"]["ranks"]["0"]),
        ]:
            program["tensors"] = {n + suffix: {"shape": s} for n, s in shapes.items()}
            program["inputs"] = [n + suffix for n in "ABCD"]
            program["outputs"] = [n + suffix for n in "PQ"]
            program["ops"] = [
                {
                    "type": "matmul",
                    "inputs": [a + suffix, b + suffix],
                    "outputs": [out + suffix],
                    "attrs": {},
                }
                for a, b, out in [("A", "B", "P"), ("C", "D", "Q")]
            ]
        raw["input_relations"] = [identity_binding(n, world=1) for n in "ABCD"]
        raw["output_relations"] = [identity_binding(n, world=1) for n in "PQ"]
        result = reduced(raw, max_contraction=2)
        self.assertEqual(result.shape_attempt.plan.block_size, 1)
        self.assertEqual(result.shape_attempt.reduced.status, Status.PROVED)
        self.assertEqual(result.status, Status.UNSUPPORTED)
        self.assertIn("contraction", result.shape_attempt.fallback_reason)

    def test_operator_alias_uses_operator_owned_shape_hooks(self):
        raw = fixture("reshape_shard_dim_0_proved")
        base = default_registry()
        registry = OpRegistry(
            {
                **{name: base.get(name) for name in base.names},
                "reshape_alias": View("reshape"),
            }
        )
        for program in [raw["single"], *raw["distributed"]["ranks"].values()]:
            program["ops"][0]["type"] = "reshape_alias"
        result = verify_stage(stage_from_dict(raw), registry, shape_reduce=True)
        self.assertEqual(result.status, Status.PROVED, result.reason)
        self.assertEqual(result.mode, "SHAPE_REDUCED_ELEMENTWISE")

    def test_generated_invalid_candidate_is_an_internal_error_not_invalid_input(self):
        with patch(
            "src.verification.shape_mode.element_proof",
            side_effect=InvalidInput("generated shape mismatch"),
        ):
            result = reduced(identity_stage((8,)))
        self.assertEqual(result.status, Status.UNKNOWN)
        self.assertEqual(result.mode, "SHAPE_REDUCTION_ERROR")
        self.assertIn("Internal", result.reason)
        self.assertIsNotNone(result.validated)
        self.assertIsNotNone(result.shape_attempt.plan)

    def test_reduced_unknown_does_not_become_proved_without_original_check(self):
        with (
            patch(
                "src.verification.shape_mode.check",
                return_value=SolverOutcome("UNKNOWN", "", 0, reason="test"),
            ),
            patch(
                "src.verification.verifier.check",
                return_value=SolverOutcome("UNKNOWN", "", 0, reason="test"),
            ),
        ):
            result = reduced(identity_stage((8,)))
        self.assertEqual(result.status, Status.UNKNOWN)
        self.assertEqual(result.mode, "SYMBOLIC_INDEX_FALLBACK")

    def test_reduced_success_is_not_sufficient_for_original_success(self):
        from src.verification.shape_mode import element_proof

        raw = identity_stage((8,))
        raw["input_relations"] = []
        with patch("src.verification.shape_mode.element_proof") as fake:
            good = identity_stage((2,))
            fake.return_value = element_proof(
                stage_from_dict(good), default_registry(), 30000
            )
            result = reduced(raw)
        self.assertEqual(result.status, Status.DISPROVED)
        self.assertEqual(result.shape_attempt.certificate.status, Status.UNKNOWN)

    def test_report_contains_two_distinct_proof_scopes_and_replayable_queries(self):
        result = reduced(fixture("mlp_dp2_tp4_03_fc2_partial"))
        text = render_report(result)
        self.assertIn("REDUCED SHAPES ONLY", text)
        self.assertIn("Original-shape certificate", text)
        self.assertIn("finite_sum", text)
        self.assertIn("Matmul M/K/N", text)
        queries = re.findall(
            r"BEGIN SMT-LIB \(complete query\)\n(.*?)\nEND SMT-LIB", text, re.S
        )
        self.assertEqual(len(queries), 4)
        for query in queries:
            solver = z3.Solver()
            solver.from_string(query)
            self.assertEqual(solver.check(), z3.unsat)

    def test_cli_switch_and_report_location(self):
        path = next(INPUT.rglob("mlp_dp2_tp4_sp_01_allgather.json"))
        self.assertNotEqual(demo_path(path), demo_path(path, True))
        with (
            tempfile.TemporaryDirectory() as tmp,
            contextlib.redirect_stdout(io.StringIO()) as stdout,
        ):
            report = Path(tmp) / "demo.txt"
            self.assertEqual(
                main(["verify", str(path), "--shape-reduce", "--report", str(report)]),
                0,
            )
            self.assertIn("SHAPE_REDUCED_ELEMENTWISE", stdout.getvalue())
            self.assertIn("Reduced Stage JSON", report.read_text())


class EncodingTests(unittest.TestCase):
    def test_finite_inputs_are_independent_reals_and_no_symbolic_lookup(self):
        context = ElementContext()
        tensor = context.input(TensorRef(None, "X"), (2, 3))
        self.assertEqual(len(context.elements[TensorRef(None, "X")]), 6)
        self.assertFalse(tensor.at((0, 0)).eq(tensor.at((0, 1))))
        self.assertFalse(context.functions)
        with self.assertRaises(Unsupported):
            tensor.at((z3.Int("i"), 0))

    def test_block_sums_share_only_identical_bodies(self):
        context = BlockContext(4)
        f = z3.Function("test_f", z3.IntSort(), z3.RealSort())
        a = context.contract(8, lambda k: f(k))
        b = context.contract(8, lambda k: f(k + 0))
        c = context.contract(8, lambda k: f(k + 1))
        self.assertTrue(z3.is_true(z3.simplify(a == b)))
        solver = z3.Solver()
        solver.add(a != c)
        self.assertEqual(solver.check(), z3.sat)
        self.assertEqual(len(context.atoms), 4)

    def test_nested_block_sums_fail_closed(self):
        context = BlockContext(2)
        with self.assertRaises(Unsupported):
            context.contract(4, lambda i: context.contract(4, lambda j: i * j))
        inner = context.contract(4, lambda i: z3.ToReal(i))
        with self.assertRaises(Unsupported):
            context.contract(4, lambda i: inner * i)
