import json
from pathlib import Path
from types import MappingProxyType

from ..core.errors import InvalidInput
from ..core.json import array, freeze, integer, mapping, names, obj, rank_key, string
from ..core.program import OpCall, Program
from ..core.tensor import TensorSpec
from ..core.topology import Topology
from ..relation.codec import decode_binding, encode_binding
from .model import StageSpec


def _program(raw, path):
    obj(raw, "tensors inputs outputs ops", path)
    tensors = {}
    for name, value in mapping(raw["tensors"], path + ".tensors").items():
        obj(value, "shape", path + "." + name)
        shape = tuple(
            integer(n, path + ".shape", 1) for n in array(value["shape"], path)
        )
        if not shape:
            raise InvalidInput("scalar tensors are outside this contract")
        tensors[name] = TensorSpec(name, shape)
    calls = []
    for i, value in enumerate(array(raw["ops"], path + ".ops")):
        op_path = f"{path}.ops[{i}]"
        obj(value, "type inputs outputs attrs", op_path)
        calls.append(
            OpCall(
                string(value["type"], op_path),
                tuple(string(n, op_path) for n in array(value["inputs"], op_path)),
                names(value["outputs"], op_path),
                freeze(mapping(value["attrs"], op_path)),
            )
        )
    return Program(
        MappingProxyType(tensors),
        names(raw["inputs"], path),
        names(raw["outputs"], path),
        tuple(calls),
    )


def stage_from_dict(raw):
    obj(
        raw,
        "name topology single distributed input_relations output_relations",
        "stage",
    )
    t = obj(raw["topology"], "world_size groups families", "topology")
    groups = {}
    for name, g in mapping(t["groups"], "groups").items():
        obj(g, "ranks", "group " + name)
        groups[name] = tuple(integer(r, "rank") for r in array(g["ranks"], "ranks"))
    families = {}
    for name, f in mapping(t["families"], "families").items():
        obj(f, "groups", "family " + name)
        families[name] = names(f["groups"], "family " + name)
    topology = Topology(
        integer(t["world_size"], "world_size", 1),
        MappingProxyType(groups),
        MappingProxyType(families),
    )
    obj(raw["distributed"], "ranks", "distributed")
    distributed = {
        rank_key(r, "rank"): _program(p, f"rank{r}")
        for r, p in mapping(raw["distributed"]["ranks"], "ranks").items()
    }
    return StageSpec(
        string(raw["name"], "name"),
        topology,
        _program(raw["single"], "single"),
        MappingProxyType(distributed),
        tuple(
            decode_binding(b, f"input_relations[{i}]")
            for i, b in enumerate(array(raw["input_relations"], "input_relations"))
        ),
        tuple(
            decode_binding(b, f"output_relations[{i}]")
            for i, b in enumerate(array(raw["output_relations"], "output_relations"))
        ),
    )


def _unique(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise InvalidInput(f"duplicate JSON key {k!r}")
        result[k] = v
    return result


def load_stage(path):
    try:
        raw = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_unique,
            parse_constant=lambda x: (_ for _ in ()).throw(
                InvalidInput(f"invalid JSON constant {x}")
            ),
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        raise InvalidInput(f"cannot load stage: {e}") from e
    return stage_from_dict(raw)


def _plain(value):
    if isinstance(value, tuple):
        return [_plain(x) for x in value]
    if hasattr(value, "items"):
        return {k: _plain(v) for k, v in value.items()}
    return value


def stage_to_dict(stage):
    def program(p):
        return {
            "tensors": {n: {"shape": list(t.shape)} for n, t in p.tensors.items()},
            "inputs": list(p.inputs),
            "outputs": list(p.outputs),
            "ops": [
                {
                    "type": o.type,
                    "inputs": list(o.inputs),
                    "outputs": list(o.outputs),
                    "attrs": _plain(o.attrs),
                }
                for o in p.ops
            ],
        }

    return {
        "name": stage.name,
        "topology": {
            "world_size": stage.topology.world_size,
            "groups": {
                n: {"ranks": list(rs)} for n, rs in stage.topology.groups.items()
            },
            "families": {
                n: {"groups": list(gs)} for n, gs in stage.topology.families.items()
            },
        },
        "single": program(stage.single),
        "distributed": {
            "ranks": {str(r): program(p) for r, p in stage.distributed.items()}
        },
        "input_relations": [encode_binding(b) for b in stage.input_relations],
        "output_relations": [encode_binding(b) for b in stage.output_relations],
    }
