"""Sound overapproximation of fixed finite sums, used only for UNSAT certificates.

An atom denotes Sum(body(q), q=0..width-1) under the current free indices.
Only syntactically equal simplified bodies share atoms. All other atoms are
independent. Every original evaluation extends to an assignment of these atoms;
therefore UNSAT implies original equivalence, but SAT is not a counterexample.
"""

import z3

from ..core.errors import Unsupported
from .context import SymbolicContext


class BlockContext(SymbolicContext):
    def __init__(self, width):
        super().__init__()
        self.width = width
        self.dummy = z3.FreshInt("bound_sum_index")
        self.atoms = {}
        self.atom_ids = set()
        self.inside_sum = False

    def contraction_terms(self, length):
        return length // self.width if length % self.width == 0 else length

    def contract(self, length, body):
        if self.width == 1 or length % self.width:
            return super().contract(length, body)
        if self.inside_sum:
            raise Unsupported(
                "nested blocked contractions require a stronger certificate"
            )
        terms = []
        for start in range(0, length, self.width):
            self.inside_sum = True
            try:
                expression = z3.simplify(body(start + self.dummy))
            finally:
                self.inside_sum = False
            # Cached inner sums also cannot be treated as independent of q.
            todo, seen = [expression], set()
            while todo:
                node = todo.pop()
                if node.get_id() in seen:
                    continue
                seen.add(node.get_id())
                if node.get_id() in self.atom_ids:
                    raise Unsupported(
                        "a blocked contraction depends on another opaque sum"
                    )
                todo.extend(node.children())
            key = (self.width, expression.sexpr())
            if key not in self.atoms:
                atom = z3.FreshReal("finite_sum")
                self.atoms[key] = atom
                self.atom_ids.add(atom.get_id())
            terms.append(self.atoms[key])
        return z3.Sum(terms)
