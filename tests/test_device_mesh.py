from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stage.mesh import (
    DeviceMeshError,
    coordinate_to_rank,
    enumerate_coordinate_group,
    mesh_size,
    rank_to_coordinate,
)
from src.stage.model import DeviceMeshSpec


def test_row_major_rank_coordinate_mapping() -> None:
    mesh = DeviceMeshSpec((2, 4))
    assert mesh_size(mesh) == 8
    assert [rank_to_coordinate(rank, mesh) for rank in range(8)] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
    ]


@pytest.mark.parametrize("shape", [(2,), (2, 3), (2, 2, 3)])
def test_rank_coordinate_round_trip_for_arbitrary_mesh_rank(
    shape: tuple[int, ...],
) -> None:
    mesh = DeviceMeshSpec(shape)
    for rank in range(mesh_size(mesh)):
        assert coordinate_to_rank(rank_to_coordinate(rank, mesh), mesh) == rank


def test_coordinate_group_varies_only_selected_axes() -> None:
    mesh = DeviceMeshSpec((2, 3, 2))
    assert enumerate_coordinate_group(mesh, (1, 0, 1), (1, 2)) == (
        (1, 0, 0),
        (1, 0, 1),
        (1, 1, 0),
        (1, 1, 1),
        (1, 2, 0),
        (1, 2, 1),
    )


@pytest.mark.parametrize(
    "mesh",
    [DeviceMeshSpec(()), DeviceMeshSpec((2, 0)), DeviceMeshSpec((2, True))],
)
def test_invalid_mesh_shapes_are_rejected(mesh: DeviceMeshSpec) -> None:
    with pytest.raises(DeviceMeshError):
        mesh_size(mesh)


def test_invalid_rank_coordinate_and_group_are_rejected() -> None:
    mesh = DeviceMeshSpec((2, 2))
    with pytest.raises(DeviceMeshError, match="rank"):
        rank_to_coordinate(4, mesh)
    with pytest.raises(DeviceMeshError, match="component"):
        coordinate_to_rank((0, 2), mesh)
    with pytest.raises(DeviceMeshError, match="duplicates"):
        enumerate_coordinate_group(mesh, (0, 0), (1, 1))
