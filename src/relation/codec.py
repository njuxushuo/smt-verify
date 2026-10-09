from types import MappingProxyType

from ..core.errors import Unsupported
from ..core.json import array, integer, mapping, obj, rank_key, string
from .model import (
    IndexPhi,
    Partial,
    RangeMap,
    RankMap,
    RelationBinding,
    RelationTerm,
    TensorRelation,
)


def decode_binding(raw, path="relation") -> RelationBinding:
    obj(raw, "single_tensor distributed_tensors relation", path)
    single = string(raw["single_tensor"], path + ".single_tensor")
    locals_ = {
        rank_key(k, path): string(v, path)
        for k, v in mapping(raw["distributed_tensors"], path).items()
    }
    obj(raw["relation"], "terms", path + ".relation")
    terms = []
    for t in array(raw["relation"]["terms"], path + ".terms"):
        obj(t, "family expr", path + ".term")
        family = string(t["family"], path + ".family")
        e = mapping(t["expr"], path + ".expr")
        if "kind" not in e:
            from ..core.errors import InvalidInput

            raise InvalidInput(f"{path}: missing expr.kind")
        if e.get("kind") == "partial":
            obj(e, "kind op", path + ".expr")
            expression = Partial(string(e["op"], path + ".op"))
        elif e.get("kind") == "index_phi":
            obj(e, "kind dim maps", path + ".expr")
            dim = integer(e["dim"], path + ".dim")
            maps = []
            for m in array(e["maps"], path + ".maps"):
                obj(m, "rank local_extent single_extent pieces", path + ".map")
                pieces = []
                for p in array(m["pieces"], path + ".pieces"):
                    obj(p, "local single", path + ".piece")
                    for key in ("local", "single"):
                        obj(p[key], "start stop", path + "." + key)
                    pieces.append(
                        RangeMap(
                            *(
                                integer(p[k][b], path + "." + k + "." + b)
                                for k in ("local", "single")
                                for b in ("start", "stop")
                            )
                        )
                    )
                maps.append(
                    RankMap(
                        integer(m["rank"], path),
                        integer(m["local_extent"], path, 1),
                        integer(m["single_extent"], path, 1),
                        tuple(pieces),
                    )
                )
            expression = IndexPhi(dim, tuple(maps))
        else:
            raise Unsupported(
                f"{path}: unsupported relation expression {e.get('kind')!r}"
            )
        terms.append(RelationTerm(family, expression))
    return RelationBinding(
        single, MappingProxyType(locals_), TensorRelation(tuple(terms))
    )


def encode_binding(binding):
    terms = []
    for term in binding.relation.terms:
        e = term.expr
        if isinstance(e, Partial):
            expr = {"kind": "partial", "op": e.op}
        else:
            expr = {
                "kind": "index_phi",
                "dim": e.dim,
                "maps": [
                    {
                        "rank": m.rank,
                        "local_extent": m.local_extent,
                        "single_extent": m.single_extent,
                        "pieces": [
                            {
                                "local": {"start": p.local_start, "stop": p.local_stop},
                                "single": {
                                    "start": p.single_start,
                                    "stop": p.single_stop,
                                },
                            }
                            for p in m.pieces
                        ],
                    }
                    for m in e.maps
                ],
            }
        terms.append({"family": term.family, "expr": expr})
    return {
        "single_tensor": binding.single_tensor,
        "distributed_tensors": {
            str(r): n for r, n in binding.distributed_tensors.items()
        },
        "relation": {"terms": terms},
    }
