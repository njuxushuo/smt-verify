import copy
from dataclasses import replace
import unittest
from unittest.mock import patch

from src.app.bootstrap import default_registry
from src.smt.solver import SolverOutcome
from src.stage.codec import stage_from_dict
from src.relation.model import TensorRelation
from src.verification.result import Status
from src.verification.verifier import verify_file, verify_stage
from tests.support import INPUT, fixture, identity_stage, phi, piece, partial


def prove(raw, **options):
    return verify_stage(stage_from_dict(raw), default_registry(), **options)


class ProofTests(unittest.TestCase):
    def test_negative_dimension_cannot_bypass_python_api_validation(self):
        raw = identity_stage((2,))
        mapping = phi(
            0,
            2,
            2,
            [[piece(0, 1, 1), piece(1, 2, 0)], [piece(0, 1, 1), piece(1, 2, 0)]],
        )
        raw["output_relations"][0]["relation"]["terms"] = [mapping]
        stage = stage_from_dict(raw)
        self.assertEqual(
            verify_stage(stage, default_registry()).status, Status.DISPROVED
        )
        binding = stage.output_relations[0]
        term = binding.relation.terms[0]
        bad = replace(
            binding,
            relation=TensorRelation((replace(term, expr=replace(term.expr, dim=-1)),)),
        )
        stage = replace(stage, output_relations=(bad,))
        self.assertEqual(
            verify_stage(stage, default_registry()).status, Status.INVALID_INPUT
        )

    def test_deep_lazy_chain_is_explicitly_unsupported(self):
        raw = identity_stage((2,), world=1)
        for rank, p in [(None, raw["single"]), *raw["distributed"]["ranks"].items()]:
            original = "X" if rank is None else "X" + rank
            previous = original
            for i in range(500):
                name = f"v{i}"
                p["tensors"][name] = {"shape": [2]}
                p["ops"].append(
                    {
                        "type": "add",
                        "inputs": [previous, original],
                        "outputs": [name],
                        "attrs": {},
                    }
                )
                previous = name
            p["outputs"] = [previous]
        b = raw["output_relations"][0]
        b["single_tensor"] = "v499"
        b["distributed_tensors"] = {"0": "v499"}
        result = prove(raw)
        self.assertEqual(result.status, Status.UNSUPPORTED)
        self.assertIn("depth", result.reason)

    def test_all_input_original_shape_obligations(self):
        for p in sorted(INPUT.rglob("*.json")):
            with self.subTest(case=p.stem):
                result = verify_file(p, default_registry())
                self.assertEqual(result.status, Status.PROVED, result.reason)
                self.assertEqual(result.scope, "ORIGINAL_SHAPES_EXACT_REAL")
                self.assertTrue(result.queries)
                self.assertTrue(all(o.status == "UNSAT" for _, o in result.queries))
                self.assertEqual(len(result.queries), len(result.reduction.domains))

    def test_missing_input_assumption_returns_actual_counterexample(self):
        raw = identity_stage()
        raw["input_relations"] = []
        result = prove(raw)
        self.assertEqual(result.status, Status.DISPROVED)
        self.assertNotEqual(
            result.counterexample["single_value"],
            result.counterexample["distributed_value"],
        )

    def test_wrong_partial_goal_is_not_assumed(self):
        raw = identity_stage()
        raw["output_relations"][0]["relation"]["terms"] = [partial()]
        self.assertEqual(prove(raw).status, Status.DISPROVED)

    def test_qkv_piecewise_maps_and_bad_output_offset(self):
        raw = identity_stage((2, 24))
        for p in raw["distributed"]["ranks"].values():
            next(iter(p["tensors"].values()))["shape"] = [2, 12]
        layout = phi(
            1,
            12,
            24,
            [
                [piece(0, 4, 0), piece(4, 8, 8), piece(8, 12, 16)],
                [piece(0, 4, 4), piece(4, 8, 12), piece(8, 12, 20)],
            ],
        )
        for f in ("input_relations", "output_relations"):
            raw[f][0]["relation"]["terms"] = [copy.deepcopy(layout)]
        self.assertEqual(prove(raw).status, Status.PROVED)
        raw["output_relations"][0]["relation"]["terms"][0]["expr"]["maps"][0]["pieces"][
            2
        ] = piece(8, 12, 20)
        result = prove(raw)
        self.assertEqual(result.status, Status.DISPROVED)
        self.assertGreaterEqual(result.counterexample["local_output_index"][1], 8)

    def test_partial_with_different_piece_order_and_same_image(self):
        raw = identity_stage((4,))
        for p in raw["distributed"]["ranks"].values():
            next(iter(p["tensors"].values()))["shape"] = [2]
        raw["topology"]["families"]["layout"] = {"groups": ["g"]}
        terms = [
            phi(
                0,
                2,
                4,
                [[piece(0, 1, 0), piece(1, 2, 2)], [piece(0, 1, 2), piece(1, 2, 0)]],
                family="layout",
            ),
            partial(),
        ]
        for f in ("input_relations", "output_relations"):
            raw[f][0]["relation"]["terms"] = copy.deepcopy(terms)
        self.assertEqual(prove(raw).status, Status.PROVED)
        raw["output_relations"][0]["relation"]["terms"][0]["expr"]["maps"][0][
            "pieces"
        ] = [piece(0, 1, 2), piece(1, 2, 0)]
        self.assertEqual(prove(raw).status, Status.DISPROVED)

    def test_partial_does_not_imply_replication(self):
        raw = identity_stage()
        raw["input_relations"][0]["relation"]["terms"] = [partial()]
        self.assertEqual(prove(raw).status, Status.DISPROVED)

    def test_original_domain_has_no_sampled_shape_lifting(self):
        raw = fixture("mlp_dp2_tp4_03_fc2_partial")
        result = prove(raw)
        self.assertEqual(result.status, Status.PROVED)
        self.assertEqual(
            result.validated.stage.single.tensors["Z"].shape, (4, 128, 1024)
        )
        self.assertEqual(result.reduction.original_output_elements, 2 * 2 * 128 * 256)
        self.assertTrue(all(d.shape == (2, 128, 256) for d in result.reduction.domains))
        self.assertEqual(prove(raw, max_contraction=128).status, Status.UNSUPPORTED)

    def test_unknown_and_invalid_are_not_proved(self):
        with patch(
            "src.verification.verifier.check",
            return_value=SolverOutcome("UNKNOWN", "", 0, reason="test timeout"),
        ):
            result = prove(identity_stage())
            self.assertEqual(result.status, Status.UNKNOWN)
        raw = identity_stage()
        raw["output_relations"] = []
        self.assertEqual(prove(raw).status, Status.INVALID_INPUT)

    def test_duplicate_operand_is_legal(self):
        raw = identity_stage()
        for rank, p in [(None, raw["single"]), *raw["distributed"]["ranks"].items()]:
            x = "X" if rank is None else "X" + rank
            y = "Y" if rank is None else "Y" + rank
            p["tensors"][y] = {"shape": [2, 3]}
            p["outputs"] = [y]
            p["ops"] = [{"type": "add", "inputs": [x, x], "outputs": [y], "attrs": {}}]
        raw["output_relations"][0]["single_tensor"] = "Y"
        raw["output_relations"][0]["distributed_tensors"] = {"0": "Y0", "1": "Y1"}
        self.assertEqual(prove(raw).status, Status.PROVED)
