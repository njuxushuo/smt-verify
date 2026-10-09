from dataclasses import dataclass
from math import prod

Shape = tuple[int, ...]


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: Shape

    @property
    def numel(self) -> int:
        return prod(self.shape)


@dataclass(frozen=True)
class TensorRef:
    rank: int | None
    name: str

    def label(self) -> str:
        return f"{'single' if self.rank is None else f'rank{self.rank}'}.{self.name}"
