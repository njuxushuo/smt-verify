from dataclasses import dataclass
from typing import Mapping

from .tensor import TensorSpec


@dataclass(frozen=True)
class OpCall:
    type: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    attrs: Mapping[str, object]


@dataclass(frozen=True)
class Program:
    tensors: Mapping[str, TensorSpec]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    ops: tuple[OpCall, ...]
