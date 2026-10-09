from ..core.errors import InvalidInput


def broadcast_shape(a, b):
    width = max(len(a), len(b))
    left = (1,) * (width - len(a)) + a
    right = (1,) * (width - len(b)) + b
    out = []
    for x, y in zip(left, right):
        if x != y and x != 1 and y != 1:
            raise InvalidInput(f"incompatible broadcast shapes {a} and {b}")
        out.append(max(x, y))
    return tuple(out)


def broadcast_index(index, shape):
    if not shape:
        return ()
    return tuple(0 if n == 1 else i for i, n in zip(index[-len(shape) :], shape))
