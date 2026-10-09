import itertools
import unittest

import z3

from src.app.bootstrap import default_registry
from src.core.errors import InvalidInput
from src.core.program import OpCall
from src.smt.context import SymbolicContext
from src.smt.tensor import SymbolicTensor


def numbered(shape):
    def value(index):
        offset = z3.IntVal(0)
        for i, n in zip(index, shape):
            offset = offset * n + i
        return z3.ToReal(offset + 1)

    return SymbolicTensor(shape, value)


class OperatorTests(unittest.TestCase):
    def evaluate(self, kind, tensors, attrs):
        call = OpCall(kind, tuple(f"x{i}" for i in range(len(tensors))), ("y",), attrs)
        return default_registry().get(kind).evaluate(call, tensors, SymbolicContext())

    def test_transpose_negative_axes_element_coordinates(self):
        tensor = numbered((2, 3, 4))
        out = self.evaluate("transpose", (tensor,), {"dim0": -1, "dim1": -2})
        self.assertEqual(out.shape, (2, 4, 3))
        for b, i, j in itertools.product(range(2), range(4), range(3)):
            self.assertTrue(
                z3.is_true(z3.simplify(out.at((b, i, j)) == tensor.at((b, j, i))))
            )

    def test_reshape_preserves_linear_index_order(self):
        tensor = numbered((2, 3, 4))
        out = self.evaluate("reshape", (tensor,), {"shape": (4, 6)})
        for i, j in itertools.product(range(4), range(6)):
            self.assertEqual(out.at((i, j)).as_long(), i * 6 + j + 1)

    def test_expand_repeats_singleton_values(self):
        tensor = numbered((2, 1, 3))
        out = self.evaluate("expand", (tensor,), {"shape": (2, 4, 3)})
        for b, i, j in itertools.product(range(2), range(4), range(3)):
            self.assertTrue(
                z3.is_true(z3.simplify(out.at((b, i, j)) == tensor.at((b, 0, j))))
            )

    def test_batched_matmul_uses_full_contraction_and_broadcast(self):
        a, b = numbered((2, 3, 4)), numbered((4, 2))
        out = self.evaluate("matmul", (a, b), {})
        for batch, i, j in itertools.product(range(2), range(3), range(2)):
            expected = sum(
                (batch * 12 + i * 4 + k + 1) * (k * 2 + j + 1) for k in range(4)
            )
            self.assertEqual(out.at((batch, i, j)).as_long(), expected)

    def test_elementwise_rank_mismatch_broadcast(self):
        a, b = numbered((2, 3, 4)), numbered((4,))
        for kind in ("add", "mul"):
            out = self.evaluate(kind, (a, b), {})
            expected = 24 + 4 if kind == "add" else 24 * 4
            self.assertEqual(out.at((1, 2, 3)).as_long(), expected)

    def test_invalid_shapes_and_attributes_are_rejected(self):
        for kind, tensors, attrs in [
            ("matmul", (numbered((2, 3)), numbered((4, 2))), {}),
            ("reshape", (numbered((2, 3)),), {"shape": (4, 2)}),
            ("expand", (numbered((2, 3)),), {"shape": (2, 4)}),
            ("transpose", (numbered((2, 3)),), {"dim0": True, "dim1": 1}),
            ("add", (numbered((2, 3)), numbered((2, 3))), {"alpha": 2}),
        ]:
            with self.subTest(kind=kind), self.assertRaises(InvalidInput):
                self.evaluate(kind, tensors, attrs)
