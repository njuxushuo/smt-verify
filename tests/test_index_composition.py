import copy
import itertools
import random
import unittest

import z3

from src.app.bootstrap import default_registry
from src.app.reporting import render_report
from src.core.errors import InvalidInput
from src.relation.encoding import forward, inverse
from src.relation.model import RangeMap
from src.relation.normalization import canonicalize_maps, compose_maps
from src.stage.codec import stage_from_dict, stage_to_dict
from src.stage.validation import validate_stage
from src.verification.result import Status
from tests.support import fixture, identity_stage, partial, phi, piece
from tests.test_proofs import prove


def evaluate(pieces, index):
    (piece_,) = [p for p in pieces if p.local_start <= index < p.local_stop]
    return piece_.single_start + index - piece_.local_start


def permutation_map(images):
    return tuple(RangeMap(i, i + 1, j, j + 1) for i, j in enumerate(images))


class PieceCompositionTests(unittest.TestCase):
    def test_piece_crosses_next_level_boundary(self):
        first = (RangeMap(0, 4, 2, 6),)
        second = (RangeMap(0, 4, 0, 4), RangeMap(4, 8, 12, 16))
        self.assertEqual(
            compose_maps(first, second), (RangeMap(0, 2, 2, 4), RangeMap(2, 4, 12, 14))
        )

    def test_only_contiguous_equal_translations_merge(self):
        identity = (RangeMap(0, 4, 0, 4),)
        contiguous = (RangeMap(2, 4, 4, 6), RangeMap(0, 2, 2, 4))
        self.assertEqual(compose_maps(identity, contiguous), (RangeMap(0, 4, 2, 6),))
        reordered = (RangeMap(2, 4, 0, 2), RangeMap(0, 2, 4, 6))
        self.assertEqual(
            compose_maps(identity, reordered),
            (RangeMap(0, 2, 4, 6), RangeMap(2, 4, 0, 2)),
        )

    def test_composition_does_not_enumerate_tensor_elements(self):
        n = 10**15
        first = (RangeMap(0, n, n, 2 * n),)
        second = (RangeMap(0, 2 * n, 3 * n, 5 * n),)
        self.assertEqual(compose_maps(first, second), (RangeMap(0, n, 4 * n, 5 * n),))

    def test_uncovered_intermediate_image_fails_closed(self):
        with self.assertRaisesRegex(InvalidInput, "does not cover"):
            compose_maps((RangeMap(0, 4, 2, 6),), (RangeMap(0, 4, 0, 4),))

    def test_random_three_level_injective_maps_match_element_oracle(self):
        rng = random.Random(20261010)
        for number in range(180):
            a, b, c, d = [
                rng.randrange(1, 5),
                rng.randrange(5, 9),
                rng.randrange(9, 13),
                rng.randrange(13, 19),
            ]
            first = list(canonicalize_maps(permutation_map(rng.sample(range(b), a))))
            second = list(canonicalize_maps(permutation_map(rng.sample(range(c), b))))
            third = list(canonicalize_maps(permutation_map(rng.sample(range(d), c))))
            for pieces in (first, second, third):
                rng.shuffle(pieces)
            frozen = tuple(tuple(p) for p in (first, second, third))
            with self.subTest(case=number):
                composed = compose_maps(compose_maps(frozen[0], frozen[1]), frozen[2])
                other = compose_maps(frozen[0], compose_maps(frozen[1], frozen[2]))
                self.assertEqual(composed, other)
                self.assertEqual(
                    [evaluate(composed, i) for i in range(a)],
                    [
                        evaluate(frozen[2], evaluate(frozen[1], evaluate(frozen[0], i)))
                        for i in range(a)
                    ],
                )
                self.assertEqual(
                    frozen, tuple(tuple(p) for p in (first, second, third))
                )


class OrderedRelationTests(unittest.TestCase):
    def validate(self, raw):
        return validate_stage(stage_from_dict(raw), default_registry())

    def test_eight_rank_cp_sp_composition_and_actual_allgather_proof(self):
        raw = fixture("dp2_cp2_sp2_zigzag_allgather")
        validated = self.validate(raw)
        expected = [0, 12, 4, 8, 0, 12, 4, 8]
        rel = validated.inputs[0]
        for rank, start in enumerate(expected):
            self.assertEqual(rel.pieces(rank, 1), (RangeMap(0, 4, start, start + 4),))
            self.assertEqual(
                rel.pieces(rank, 0),
                (RangeMap(0, 2, 2 * (rank // 4), 2 * (rank // 4) + 2),),
            )
            for index in itertools.product(range(2), range(4), range(8)):
                symbolic = tuple(z3.IntVal(i) for i in index)
                global_ = forward(rel, rank, symbolic)
                back = inverse(rel, rank, global_)
                self.assertEqual(tuple(z3.simplify(i).as_long() for i in back), index)
        result = prove(raw)
        self.assertEqual(result.status, Status.PROVED, result.reason)
        self.assertEqual(len(result.queries), 8)
        self.assertEqual(stage_to_dict(result.validated.stage), raw)
        report = render_report(result)
        self.assertIn("dim 1: sp -> cp", report)
        self.assertIn("level=1/2", report)
        self.assertIn("level extents 4 -> 8", report)
        self.assertIn("level extents 8 -> 16", report)
        self.assertIn("rank 1: local [0,4) -> single [12,16)", report)
        self.assertIn("intermediate [4,8)", report)

    def test_original_four_rank_example(self):
        raw = fixture("dp2_cp2_sp2_zigzag_allgather")
        raw["topology"]["world_size"] = 4
        for kind in ("sp", "cp"):
            raw["topology"]["families"][kind]["groups"] = [f"{kind}_0", f"{kind}_1"]
        raw["topology"]["families"]["tp"]["groups"] = ["sp_0", "sp_1"]
        del raw["topology"]["families"]["dp"]
        raw["topology"]["groups"] = {
            g: v
            for g, v in raw["topology"]["groups"].items()
            if g in ("sp_0", "sp_1", "cp_0", "cp_1")
        }
        raw["single"]["tensors"]["X"]["shape"][0] = 1
        raw["distributed"]["ranks"] = {
            r: p for r, p in raw["distributed"]["ranks"].items() if int(r) < 4
        }
        for p in raw["distributed"]["ranks"].values():
            for t in p["tensors"].values():
                t["shape"][0] = 1
        for section in ("input_relations", "output_relations"):
            b = raw[section][0]
            b["distributed_tensors"] = {
                r: n for r, n in b["distributed_tensors"].items() if int(r) < 4
            }
            b["relation"]["terms"] = [
                t for t in b["relation"]["terms"] if t["family"] != "dp"
            ]
            for term in b["relation"]["terms"]:
                term["expr"]["maps"] = [
                    m for m in term["expr"]["maps"] if m["rank"] < 4
                ]
        self.assertEqual(prove(raw).status, Status.PROVED)

    def test_legal_but_wrong_output_mapping_is_disproved(self):
        raw = fixture("dp2_cp2_sp2_zigzag_allgather")
        cp = raw["output_relations"][0]["relation"]["terms"][1]["expr"]
        cp["maps"][0]["pieces"][1]["single"] = {"start": 8, "stop": 12}
        self.validate(raw)
        result = prove(raw)
        self.assertEqual(result.status, Status.DISPROVED)
        self.assertGreaterEqual(result.counterexample["local_output_index"][1], 4)

    def test_extent_chain_and_endpoints_are_checked_on_every_rank(self):
        for change, message in [
            ("first", "actual local shape"),
            ("middle", "preceding level target extent"),
            ("last", "actual single shape"),
        ]:
            raw = fixture("dp2_cp2_sp2_zigzag_allgather")
            terms = raw["input_relations"][0]["relation"]["terms"]
            if change == "first":
                m = terms[1]["expr"]["maps"][7]
                m["local_extent"] = 3
                m["pieces"] = [piece(0, 3, 4)]
            elif change == "middle":
                m = terms[2]["expr"]["maps"][7]
                m["local_extent"] = 7
                m["pieces"] = [piece(0, 7, 4)]
            else:
                terms[2]["expr"]["maps"][7]["single_extent"] = 17
            with (
                self.subTest(change=change),
                self.assertRaisesRegex(InvalidInput, f"rank 7, dim 1.*{message}"),
            ):
                self.validate(raw)

    def test_every_level_checks_coverage_bounds_and_injectivity(self):
        for change in ("gap", "overlap", "duplicate_image", "bounds", "missing_rank"):
            raw = fixture("dp2_cp2_sp2_zigzag_allgather")
            e = raw["input_relations"][0]["relation"]["terms"][2]["expr"]
            m = e["maps"][4]
            if change == "gap":
                m["pieces"] = [piece(0, 3, 0), piece(4, 8, 12)]
            elif change == "overlap":
                m["pieces"] = [piece(0, 5, 0), piece(4, 8, 12)]
            elif change == "duplicate_image":
                m["pieces"] = [piece(0, 4, 0), piece(4, 8, 0)]
            elif change == "bounds":
                m["pieces"] = [piece(0, 4, 0), piece(4, 8, 13)]
            else:
                e["maps"].pop()
            with self.subTest(change=change), self.assertRaises(InvalidInput):
                self.validate(raw)

    def test_reverse_chain_is_rejected_but_other_dim_can_interleave(self):
        raw = fixture("dp2_cp2_sp2_zigzag_allgather")
        dp, sp, cp = raw["input_relations"][0]["relation"]["terms"]
        original = self.validate(raw).inputs[0]
        raw["input_relations"][0]["relation"]["terms"] = [sp, dp, cp]
        self.assertEqual(self.validate(raw).inputs[0].axes, original.axes)
        self.assertEqual(prove(raw).status, Status.PROVED)
        raw["input_relations"][0]["relation"]["terms"] = [dp, cp, sp]
        with self.assertRaisesRegex(InvalidInput, "actual local shape"):
            self.validate(raw)

    def test_equal_extent_noncommuting_permutations_obey_list_order(self):
        raw = identity_stage((3,))
        raw["topology"]["families"]["second"] = {"groups": ["g"]}
        a = phi(0, 3, 3, [[piece(0, 1, 1), piece(1, 2, 0), piece(2, 3, 2)]] * 2)
        b = phi(
            0,
            3,
            3,
            [[piece(0, 1, 0), piece(1, 2, 2), piece(2, 3, 1)]] * 2,
            family="second",
        )
        for field in ("input_relations", "output_relations"):
            raw[field][0]["relation"]["terms"] = copy.deepcopy([a, b])
        self.assertEqual(prove(raw).status, Status.PROVED)
        raw["output_relations"][0]["relation"]["terms"].reverse()
        self.validate(raw)
        self.assertEqual(prove(raw).status, Status.DISPROVED)

    def test_chain_can_compose_to_identity(self):
        raw = identity_stage((2,))
        raw["topology"]["families"]["second"] = {"groups": ["g"]}
        a = phi(0, 2, 2, [[piece(0, 1, 1), piece(1, 2, 0)]] * 2)
        b = copy.deepcopy(a)
        b["family"] = "second"
        raw["input_relations"][0]["relation"]["terms"] = [a, b]
        self.assertEqual(
            self.validate(raw).inputs[0].pieces(0, 0), (RangeMap(0, 2, 0, 2),)
        )
        self.assertEqual(prove(raw).status, Status.PROVED)

    def test_rank_specific_intermediate_extents_and_uneven_local_shapes(self):
        raw = identity_stage((8,))
        raw["topology"]["families"]["outer"] = {"groups": ["g"]}
        for rank, extent in enumerate((2, 3)):
            raw["distributed"]["ranks"][str(rank)]["tensors"][f"X{rank}"]["shape"] = [
                extent
            ]
        first = phi(0, [2, 3], 4, [[piece(0, 2, 1)], [piece(0, 3, 1)]])
        first["expr"]["maps"][1]["single_extent"] = 6
        second = phi(0, [4, 6], 8, [[piece(0, 4, 2)], [piece(0, 6, 1)]], family="outer")
        raw["input_relations"][0]["relation"]["terms"] = [first, second]
        raw["output_relations"][0]["relation"]["terms"] = [
            phi(0, [2, 3], 8, [[piece(0, 2, 3)], [piece(0, 3, 2)]])
        ]
        normalized = self.validate(raw).inputs[0]
        self.assertEqual(normalized.pieces(0, 0), (RangeMap(0, 2, 3, 5),))
        self.assertEqual(normalized.pieces(1, 0), (RangeMap(0, 3, 2, 5),))
        self.assertEqual(prove(raw).status, Status.PROVED)

    def test_three_level_chain_with_partial_and_different_local_orders(self):
        raw = identity_stage((7,))
        for p in raw["distributed"]["ranks"].values():
            next(iter(p["tensors"].values()))["shape"] = [2]
        for f in ("first", "second", "third"):
            raw["topology"]["families"][f] = {"groups": ["g"]}
        a = phi(
            0,
            2,
            4,
            [[piece(0, 1, 0), piece(1, 2, 3)], [piece(0, 1, 3), piece(1, 2, 0)]],
            family="first",
        )
        b = phi(0, 4, 6, [[piece(0, 2, 0), piece(2, 4, 4)]] * 2, family="second")
        c = phi(0, 6, 7, [[piece(0, 6, 1)]] * 2, family="third")
        # Partial is a value constraint, not an intermediate coordinate stage.
        for field in ("input_relations", "output_relations"):
            raw[field][0]["relation"]["terms"] = copy.deepcopy([a, partial(), b, c])
        rel = self.validate(raw).inputs[0]
        self.assertEqual(rel.image(0), (((1, 2), (6, 7)),))
        self.assertEqual(rel.image(0), rel.image(1))
        self.assertEqual(prove(raw).status, Status.PROVED)
        raw["output_relations"][0]["relation"]["terms"][0]["expr"]["maps"][0][
            "pieces"
        ] = [piece(0, 1, 3), piece(1, 2, 0)]
        self.assertEqual(prove(raw).status, Status.DISPROVED)

    def test_partial_requires_matching_composed_images(self):
        raw = identity_stage((6,))
        for p in raw["distributed"]["ranks"].values():
            next(iter(p["tensors"].values()))["shape"] = [2]
        for f in ("first", "second"):
            raw["topology"]["families"][f] = {"groups": ["g"]}
        a = phi(0, 2, 4, [[piece(0, 2, 0)], [piece(0, 2, 2)]], family="first")
        b = phi(0, 4, 6, [[piece(0, 4, 1)]] * 2, family="second")
        raw["input_relations"][0]["relation"]["terms"] = [a, b, partial()]
        with self.assertRaisesRegex(InvalidInput, "same mapped single domain"):
            self.validate(raw)
