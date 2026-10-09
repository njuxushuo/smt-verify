from ..ops.api import ExecutionLimits
from ..ops.collectives import Collective
from ..ops.elementwise import Elementwise
from ..ops.matmul import MatMul
from ..ops.registry import OpRegistry
from ..ops.view import View


def default_registry(limits=ExecutionLimits()):
    operators = {"matmul": MatMul(limits)}
    operators.update({name: Elementwise(name) for name in ("add", "mul", "gelu")})
    operators.update({name: View(name) for name in ("transpose", "reshape", "expand")})
    operators.update(
        {
            name: Collective(name)
            for name in ("all_gather", "all_reduce", "reduce_scatter")
        }
    )
    return OpRegistry(operators)
