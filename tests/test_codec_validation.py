import copy
import json
import tempfile
import unittest
from pathlib import Path

from src.app.bootstrap import default_registry
from src.core.errors import InvalidInput, Unsupported
from src.stage.codec import load_stage, stage_from_dict, stage_to_dict
from src.stage.validation import validate_stage
from tests.support import (
    INPUT,
    INPUT_CASE_COUNT,
    fixture,
    identity_stage,
    phi,
    piece,
    partial,
)


class CodecValidationTests(unittest.TestCase):
    def check(self, raw):
        return validate_stage(stage_from_dict(raw), default_registry())

    def test_all_fixture_roundtrips_and_normalization(self):
        paths = list(INPUT.rglob("*.json"))
        self.assertEqual(len(paths), INPUT_CASE_COUNT)
        for path in paths:
            with self.subTest(path=path):
                stage = load_stage(path)
                self.assertEqual(stage_to_dict(stage), json.loads(path.read_text()))
                self.assertTrue(validate_stage(stage, default_registry()).checks)

    def test_strict_unknown_fields_and_duplicate_json_keys(self):
        for key in ("schema", "semantics", "extra"):
            raw = identity_stage()
            raw[key] = 1
            with self.assertRaises(InvalidInput):
                stage_from_dict(raw)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.json"
            p.write_text('{"name":"a","name":"b"}')
            with self.assertRaisesRegex(InvalidInput, "duplicate"):
                load_stage(p)

    def test_missing_rank_boolean_shape_and_noncanonical_rank(self):
        raw = identity_stage()
        del raw["distributed"]["ranks"]["1"]
        with self.assertRaises(InvalidInput):
            self.check(raw)
        raw = identity_stage()
        raw["single"]["tensors"]["X"]["shape"][0] = True
        with self.assertRaises(InvalidInput):
            self.check(raw)
        raw = identity_stage()
        raw["distributed"]["ranks"]["01"] = raw["distributed"]["ranks"].pop("1")
        with self.assertRaises(InvalidInput):
            self.check(raw)

    def test_missing_output_goal_and_produced_input_binding(self):
        raw = identity_stage()
        raw["output_relations"] = []
        with self.assertRaises(InvalidInput):
            self.check(raw)
        raw = fixture("add_hidden_sharded_bias_to_shard")
        raw["input_relations"][0] = copy.deepcopy(raw["output_relations"][0])
        with self.assertRaisesRegex(InvalidInput, "boundary"):
            self.check(raw)

    def test_invalid_intervals_extents_and_rank_maps(self):
        baseline = fixture("mlp_dp2_tp4_sp_01_allgather")
        for mutation in ("gap", "overlap", "extent", "missing", "identity"):
            raw = copy.deepcopy(baseline)
            expr = raw["input_relations"][0]["relation"]["terms"][0]["expr"]
            if mutation == "gap":
                expr["maps"][0]["pieces"][0]["local"]["start"] = 1
            if mutation == "overlap":
                expr["maps"][0]["pieces"] *= 2
            if mutation == "extent":
                expr["maps"][0]["single_extent"] += 1
            if mutation == "missing":
                expr["maps"].pop()
            if mutation == "identity":
                expr["maps"] = []
            with self.subTest(mutation=mutation), self.assertRaises(InvalidInput):
                self.check(raw)

    def test_partial_image_mismatch_is_rejected(self):
        raw = identity_stage((4,))
        for pg in raw["distributed"]["ranks"].values():
            next(iter(pg["tensors"].values()))["shape"] = [2]
        raw["topology"]["families"]["layout"] = {"groups": ["g"]}
        terms = [
            phi(0, 2, 4, [[piece(0, 2, 0)], [piece(0, 2, 2)]], family="layout"),
            partial(),
        ]
        for field in ("input_relations", "output_relations"):
            raw[field][0]["relation"]["terms"] = copy.deepcopy(terms)
        with self.assertRaisesRegex(InvalidInput, "same mapped"):
            self.check(raw)

    def test_unknown_operator_is_unsupported_and_invalid_index_chain_is_rejected(self):
        raw = fixture("mlp_dp2_tp4_02_fc1_gelu")
        raw["single"]["ops"][0]["type"] = "not_implemented"
        with self.assertRaises(Unsupported):
            self.check(raw)
        raw = fixture("mlp_dp2_tp4_sp_01_allgather")
        terms = raw["input_relations"][0]["relation"]["terms"]
        terms[1]["expr"]["dim"] = 0
        with self.assertRaisesRegex(InvalidInput, "preceding level target extent"):
            self.check(raw)

    def test_empty_terms_are_full_identity_not_missing_relation(self):
        normalized = self.check(identity_stage()).inputs[0]
        self.assertEqual(normalized.axes, {})
        self.assertEqual(normalized.image(0), (((0, 2),), ((0, 3),)))
