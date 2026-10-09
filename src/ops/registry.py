from ..core.errors import Unsupported
from .api import CollectiveOperator, LocalOperator


class OpRegistry:
    def __init__(self, operators: dict[str, LocalOperator | CollectiveOperator]):
        self._operators = dict(operators)

    def get(self, name: str) -> LocalOperator | CollectiveOperator:
        if name not in self._operators:
            raise Unsupported(f"operator {name!r} is not implemented")
        return self._operators[name]

    @property
    def names(self):
        return tuple(sorted(self._operators))
