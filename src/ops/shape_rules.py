"""Operator-owned constraints for the optional integral-block reduction policy."""

from math import prod

from ..core.errors import Unsupported


def broadcast(constraints, source, original, inputs, output):
    width = len(output)
    padded = [(1,) * (width - len(s)) + s for s in original]
    values = [(1,) * (width - len(s)) + s for s in inputs]
    for axis in range(width):
        for old, new in zip(padded, values):
            constraints.add(
                source,
                "Preserve original broadcast axis role.",
                new[axis] == (1 if old[axis] == 1 else output[axis]),
            )


def matmul(c, source, call, original, inputs, output, group_size):
    a, b = inputs
    c.add(
        source,
        "Matmul M/K/N alignment and nondegenerate contraction.",
        a[-1] == b[-2],
        output[-2] == a[-2],
        output[-1] == b[-1],
        a[-1] >= min(2, original[0][-1]),
    )
    broadcast(
        c, source, [s[:-2] for s in original], [s[:-2] for s in inputs], output[:-2]
    )


def elementwise(c, source, call, original, inputs, output, group_size):
    broadcast(c, source, original, inputs, output)


def view(kind, c, source, call, original, inputs, output, group_size):
    (x,) = inputs
    if kind == "transpose":
        axes = list(range(len(x)))
        a, b = (call.attrs[k] % len(x) for k in ("dim0", "dim1"))
        axes[a], axes[b] = axes[b], axes[a]
        c.add(
            source,
            "Preserve transpose axes.",
            *(output[d] == x[a] for d, a in enumerate(axes)),
        )
    elif kind == "expand":
        broadcast(c, source, original, inputs, output)
    else:
        # Match contiguous row-major axis groups by their ORIGINAL products.
        old_in, old_out = original[0], call.attrs["shape"]
        i = j = 0
        while i < len(x) and j < len(output):
            left, right = [i], [j]
            p, q = old_in[i], old_out[j]
            while p != q:
                if p < q:
                    i += 1
                    if i == len(x):
                        raise Unsupported("reshape axis grouping is unsupported")
                    left.append(i)
                    p *= old_in[i]
                else:
                    j += 1
                    if j == len(output):
                        raise Unsupported("reshape axis grouping is unsupported")
                    right.append(j)
                    q *= old_out[j]
            c.add(
                source,
                f"Row-major reshape group {left} -> {right}.",
                prod(x[d] for d in left) == prod(output[d] for d in right),
            )
            i += 1
            j += 1
        c.add(source, "Reshape total element count.", prod(x) == prod(output))


def collective(kind, c, source, call, original, inputs, output, group_size):
    (x,) = inputs
    axis = call.attrs.get("axis")
    for d in range(len(x)):
        if d != axis or kind == "all_reduce":
            equation = output[d] == x[d]
        elif kind == "all_gather":
            equation = output[d] == group_size * x[d]
        else:
            equation = x[d] == group_size * output[d]
        c.add(
            source, "Collective length and unchanged noncommunication axes.", equation
        )
