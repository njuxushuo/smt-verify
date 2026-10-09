import unittest

from src.verification.result import Status
from tests.support import fixture, identity_stage
from tests.test_proofs import prove


class CollectiveTests(unittest.TestCase):
    def test_cross_group_dependency_deadlock_is_rejected(self):
        raw = identity_stage((2,))
        raw["topology"]["groups"]["other"] = {"ranks": [0, 1]}
        for rank, p in raw["distributed"]["ranks"].items():
            first, second = ("g", "other") if rank == "0" else ("other", "g")
            p["tensors"].update(
                {f"A{rank}": {"shape": [2]}, f"B{rank}": {"shape": [2]}}
            )
            p["ops"] = [
                {
                    "type": "all_reduce",
                    "inputs": [f"X{rank}"],
                    "outputs": [f"A{rank}"],
                    "attrs": {"group": first, "reduce_op": "sum"},
                },
                {
                    "type": "all_reduce",
                    "inputs": [f"A{rank}"],
                    "outputs": [f"B{rank}"],
                    "attrs": {"group": second, "reduce_op": "sum"},
                },
            ]
            p["outputs"] = [f"B{rank}"]
            raw["output_relations"][0]["distributed_tensors"][rank] = f"B{rank}"
        result = prove(raw)
        self.assertEqual(result.status, Status.INVALID_INPUT)
        self.assertIn("cannot make progress", result.reason)

    def test_group_order_changes_allgather_values(self):
        raw = fixture("mlp_dp2_tp4_sp_01_allgather")
        raw["topology"]["groups"]["tp_0"]["ranks"].reverse()
        self.assertEqual(prove(raw).status, Status.DISPROVED)

    def test_group_order_changes_reduce_scatter_assignment(self):
        raw = fixture("mlp_dp2_tp4_sp_04_reduce_scatter")
        raw["topology"]["groups"]["tp_1"]["ranks"].reverse()
        self.assertEqual(prove(raw).status, Status.DISPROVED)

    def test_reordering_allreduce_members_does_not_change_sum(self):
        raw = fixture("mlp_dp2_tp4_04_allreduce")
        raw["topology"]["groups"]["tp_1"]["ranks"].reverse()
        self.assertEqual(prove(raw).status, Status.PROVED)

    def test_missing_collective_member_is_invalid(self):
        raw = fixture("mlp_dp2_tp4_04_allreduce")
        p = raw["distributed"]["ranks"]["3"]
        p["ops"] = []
        p["inputs"].append("Y3")
        self.assertEqual(prove(raw).status, Status.INVALID_INPUT)

    def test_wrong_group_membership_is_invalid(self):
        raw = fixture("mlp_dp2_tp4_04_allreduce")
        raw["distributed"]["ranks"]["0"]["ops"][0]["attrs"]["group"] = "tp_1"
        self.assertEqual(prove(raw).status, Status.INVALID_INPUT)

    def test_no_cross_dp_reduction(self):
        raw = fixture("mlp_dp2_tp4_04_allreduce")
        raw["topology"]["groups"]["world"] = {"ranks": list(range(8))}
        for p in raw["distributed"]["ranks"].values():
            p["ops"][0]["attrs"]["group"] = "world"
        self.assertEqual(prove(raw).status, Status.DISPROVED)

    def test_multiple_collectives_pair_by_occurrence(self):
        raw = fixture("mlp_dp2_tp4_04_allreduce")
        for rank, p in raw["distributed"]["ranks"].items():
            p["tensors"]["Q" + rank] = {"shape": [2, 128, 256]}
            p["ops"].append(
                {
                    "type": "all_reduce",
                    "inputs": ["Y" + rank],
                    "outputs": ["Q" + rank],
                    "attrs": {"group": "tp_" + str(int(rank) // 4), "reduce_op": "sum"},
                }
            )
            p["outputs"] = ["Q" + rank]
            raw["output_relations"][0]["distributed_tensors"][rank] = "Q" + rank
        # Reducing a replicated sum a second time scales it by TP degree.
        self.assertEqual(prove(raw).status, Status.DISPROVED)
