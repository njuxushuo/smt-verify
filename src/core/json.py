"""Small strict JSON helpers shared by independent codecs."""

from types import MappingProxyType

from .errors import InvalidInput


def obj(value, fields: str, path: str):
    if not isinstance(value, dict) or set(value) != set(fields.split()):
        raise InvalidInput(f"{path}: expected exactly fields [{fields}]")
    return value


def mapping(value, path):
    if not isinstance(value, dict):
        raise InvalidInput(f"{path}: expected object")
    for k in value:
        string(k, path + " key")
    return value


def array(value, path):
    if not isinstance(value, list):
        raise InvalidInput(f"{path}: expected array")
    return value


def integer(value, path, minimum=0):
    if type(value) is not int or value < minimum:
        raise InvalidInput(f"{path}: expected integer >= {minimum}")
    return value


def string(value, path):
    if not isinstance(value, str) or not value:
        raise InvalidInput(f"{path}: expected nonempty string")
    return value


def names(value, path):
    result = tuple(string(x, path) for x in array(value, path))
    if len(set(result)) != len(result):
        raise InvalidInput(f"{path}: duplicate names")
    return result


def rank_key(value, path):
    string(value, path)
    if not value.isascii() or not value.isdecimal() or str(int(value)) != value:
        raise InvalidInput(f"{path}: rank key must be canonical nonnegative decimal")
    return int(value)


def freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(freeze(v) for v in value)
    return value
