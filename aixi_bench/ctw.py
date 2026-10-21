"""Context Tree Weighting (Willems, Shtarkov, Tjalkens) over a binary alphabet.

Every node holds Krichevsky-Trofimov (KT) counts and the log KT block probability; the
weighted probability is computed recursively at every node:

    Pw(node) = 1/2 * Pe(node) + 1/2 * Pw(child0) * Pw(child1)      (internal nodes)
    Pw(leaf) = Pe(leaf)

Updates are reversible: ``update`` returns an undo record and ``revert`` restores the exact
previous tree, so planning code can condition on hypothetical bits without mutating the model.
"""
import math
from hashlib import sha256

LOG_HALF = math.log(0.5)


def _logaddexp(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


class CTW:
    def __init__(self, depth: int):
        self.depth = depth
        self.nodes = {}  # context suffix (most recent bit first) -> [n0, n1, log_pe, log_pw]

    def root_log_pw(self) -> float:
        node = self.nodes.get(())
        return 0.0 if node is None else node[3]

    def update(self, ctx, bit: int):
        """Update every node on the context path with ``bit``; return an undo record.

        ``ctx`` lists the preceding bits, most recent first, with at least ``depth`` entries.
        """
        depth = self.depth
        path = [tuple(ctx[:d]) for d in range(depth + 1)]
        undo = []
        for key in path:
            node = self.nodes.get(key)
            undo.append((key, None if node is None else node[:]))
            if node is None:
                node = self.nodes[key] = [0, 0, 0.0, 0.0]
            node[2] += math.log((node[bit] + 0.5) / (node[0] + node[1] + 1.0))
            node[bit] += 1
        for d in range(depth, -1, -1):
            node = self.nodes[path[d]]
            if d == depth:
                node[3] = node[2]
            else:
                lp = 0.0
                for b in (0, 1):
                    child = self.nodes.get(path[d] + (b,))
                    if child is not None:
                        lp += child[3]
                node[3] = _logaddexp(LOG_HALF + node[2], LOG_HALF + lp)
        return undo

    def revert(self, undo):
        for key, old in reversed(undo):
            if old is None:
                del self.nodes[key]
            else:
                self.nodes[key] = old

    def prob_bit(self, ctx, bit: int) -> float:
        """P(bit | ctx) under the CTW mixture, computed as a ratio of block probabilities."""
        before = self.root_log_pw()
        undo = self.update(ctx, bit)
        after = self.root_log_pw()
        self.revert(undo)
        return math.exp(after - before)

    def digest(self) -> str:
        return sha256(repr(sorted(self.nodes.items())).encode("utf-8")).hexdigest()
