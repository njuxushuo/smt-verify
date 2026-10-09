from dataclasses import dataclass
from typing import Protocol

from ..core.errors import InvalidInput
from ..core.program import OpCall
from ..core.tensor import Shape
from ..smt.context import SymbolicContext
from ..smt.tensor import SymbolicTensor


def attrs_exact(call, required=()):
    if set(call.attrs) != set(required):
        raise InvalidInput(f"{call.type}: attrs must contain exactly {tuple(required)}")


def arity(call, inputs):
    if len(call.inputs) != inputs or len(call.outputs) != 1:
        raise InvalidInput(f"{call.type}: expected {inputs} inputs and one output")


def axis(value, rank):
    if type(value) is not int or not -rank <= value < rank:
        raise InvalidInput(f"axis {value!r} is outside tensor rank {rank}")
    return value % rank


class Operator(Protocol):
    collective: bool

    def infer(
        self, call: OpCall, input_shapes: tuple[Shape, ...], group_size: int = 1
    ) -> Shape: ...


class LocalOperator(Operator, Protocol):
    def evaluate(
        self, call: OpCall, inputs: tuple[SymbolicTensor, ...], context: SymbolicContext
    ) -> SymbolicTensor: ...


class CollectiveOperator(Operator, Protocol):
    def evaluate_group(
        self, call: OpCall, inputs: tuple[SymbolicTensor, ...], context: SymbolicContext
    ) -> tuple[SymbolicTensor, ...]: ...


@dataclass(frozen=True)
class ExecutionLimits:
    max_contraction: int = 8192
