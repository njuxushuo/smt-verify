import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "input"


def fixture(stem):
    (path,) = INPUT.rglob(stem + ".json")
    return json.loads(path.read_text())


def tensor(shape):
    return {"shape": list(shape)}


def identity_binding(name="X", world=2, terms=()):
    return {
        "single_tensor": name,
        "distributed_tensors": {str(r): f"{name}{r}" for r in range(world)},
        "relation": {"terms": copy.deepcopy(list(terms))},
    }


def identity_stage(shape=(2, 3), world=2):
    binding = identity_binding(world=world)
    return {
        "name": "identity",
        "topology": {
            "world_size": world,
            "groups": {"g": {"ranks": list(range(world))}},
            "families": {"tp": {"groups": ["g"]}},
        },
        "single": {
            "tensors": {"X": tensor(shape)},
            "inputs": ["X"],
            "outputs": ["X"],
            "ops": [],
        },
        "distributed": {
            "ranks": {
                str(r): {
                    "tensors": {f"X{r}": tensor(shape)},
                    "inputs": [f"X{r}"],
                    "outputs": [f"X{r}"],
                    "ops": [],
                }
                for r in range(world)
            }
        },
        "input_relations": [copy.deepcopy(binding)],
        "output_relations": [copy.deepcopy(binding)],
    }


def piece(local_start, local_stop, single_start):
    return {
        "local": {"start": local_start, "stop": local_stop},
        "single": {
            "start": single_start,
            "stop": single_start + local_stop - local_start,
        },
    }


def phi(dim, local, single, pieces_by_rank, family="tp"):
    return {
        "family": family,
        "expr": {
            "kind": "index_phi",
            "dim": dim,
            "maps": [
                {
                    "rank": r,
                    "local_extent": local[r] if isinstance(local, list) else local,
                    "single_extent": single,
                    "pieces": pieces,
                }
                for r, pieces in enumerate(pieces_by_rank)
            ],
        },
    }


def partial(family="tp"):
    return {"family": family, "expr": {"kind": "partial", "op": "sum"}}
