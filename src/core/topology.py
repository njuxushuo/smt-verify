from dataclasses import dataclass
from typing import Mapping

from .errors import InvalidInput


@dataclass(frozen=True)
class Topology:
    world_size: int
    groups: Mapping[str, tuple[int, ...]]
    families: Mapping[str, tuple[str, ...]]

    def validate(self) -> None:
        if type(self.world_size) is not int or self.world_size < 1:
            raise InvalidInput("topology.world_size must be a positive integer")
        world = set(range(self.world_size))
        for name, members in self.groups.items():
            if not members or any(type(r) is not int for r in members):
                raise InvalidInput(f"group {name}: ranks must be nonempty integers")
            if len(set(members)) != len(members) or not set(members) <= world:
                raise InvalidInput(f"group {name}: duplicate or out-of-world ranks")
        for name, ids in self.families.items():
            if (
                not ids
                or len(set(ids)) != len(ids)
                or not set(ids) <= self.groups.keys()
            ):
                raise InvalidInput(f"family {name}: invalid group references")
            members = [r for group in ids for r in self.groups[group]]
            if sorted(members) != list(range(self.world_size)):
                raise InvalidInput(
                    f"family {name}: groups must partition the whole world"
                )

    def family_groups(self, family: str) -> tuple[tuple[int, ...], ...]:
        if family not in self.families:
            raise InvalidInput(f"unknown family {family!r}")
        return tuple(self.groups[g] for g in self.families[family])
