from collections.abc import Collection, Iterable, Sequence
from functools import reduce
import operator
from typing import Literal, cast


Bin = Literal[0, 1]

DEBUG = False
old_print = print


def debug_print(*args, **kwargs):
    if DEBUG:
        old_print(*args, **kwargs)


print = debug_print


def argsorted(lst: list) -> list:
    return sorted(range(len(lst)), key=lambda i: lst[i])


class Factor:
    """We can represet a factor 𝜙 as a tuple (𝑉, <, 𝜎), where
        - 𝑉 ⊆ 𝒱 is the scope of the factor,
        - < is a total ordering of 𝒱, and
        - 𝜎 : [2^|𝑉|] → ℝ is a value function ([𝑛] = {0, 1, ..., 𝑛-1}).
    We can then induce an index function 𝑖 : 𝑉 → [|𝑉|] using the ordering.
    That is, 𝑖 can be seen as a sort function of 𝑉 given <.
    In our implementation, variables are represented as integers, so < is the usual integer ordering.
    """

    def __init__(self, scope: Collection[int], values: Sequence[float]):
        """Initializes a factor with a given scopee and values.

        @param scope: The scope of the factor, 𝑉.
        @param values: The values of the factor, 𝜎. len(values) == 2 ** len(arguments).
            Must be in the order of the binary representation of the arguments.
        """
        assert len(values) == 1 << len(scope)
        # 𝑉
        self.scope = sorted(scope)
        # 𝑖
        self._idx = {v: i for i, v in enumerate(scope)}
        # 𝑖^(-1)
        self._rev_idx = self.scope
        # 𝜎
        self._entries = list(values)

    @classmethod
    def from_cpt(cls, x: int, parents: Sequence[int], cpt: Sequence[float]) -> "Factor":
        "Creates a factor from a CPT."
        # CPT of X lacks the entries for X=0, so we fill them
        values = [v for cpt_v in cpt for v in (1 - cpt_v, cpt_v)]
        vars = [*parents, x]

        # Check if sorted; otherwise, sort the vars and the values
        # This is efficient enough for small number of parents
        sorted_vars = sorted(vars)
        if sorted_vars == vars:
            return cls(vars, values)

        i_map = argsorted(argsorted(vars))
        cnt = len(vars)
        sorted_values = [0.0] * len(values)
        for i, v in enumerate(values):
            bm = i
            j = 0
            k = 0
            while bm:
                j |= (bm & 1) << (cnt - i_map[cnt - k - 1] - 1)
                bm >>= 1
                k += 1
            sorted_values[j] = v

        return cls(sorted_vars, sorted_values)

    def __repr__(self) -> str:
        header = f"\n  {''.join(map(str, self.scope))}:"
        lines = "\n  ".join(
            f"{x:0{len(self.scope)}b}: {v:.3f}" for x, v in enumerate(self._entries)
        )
        return f"Factor({header}\n  {lines})"

    @staticmethod
    def _index(values: Sequence[int]) -> int:
        return sum(e << i for i, e in enumerate(reversed(values)))

    def __getitem__(self, values: int | tuple[int, ...]) -> float:
        """Evaluates the factor for given values in an ergonomic way.
        Useful for debugging.
        """
        if not isinstance(values, tuple):
            values = (values,)
        assert len(values) == len(self.scope)

        return self._entries[self._index(values)]

    def __mul__(self, other: "Factor") -> "Factor":
        scope = {*self.scope, *other.scope}
        cnt = len(scope)
        # Temporarily set the values to 0
        factor = Factor(scope, [0.0] * (1 << cnt))

        # Set the values for all indices; accesses an "unsafe" _entries
        for i in range(1 << cnt):
            bm = i
            # Calculate the argument bitmaps for 𝜎's of self and the other.
            bms = [0, 0]
            for j in range(-1, -cnt - 1, -1):
                x = factor.scope[j]
                flag = bm & 1
                bm >>= 1
                if not flag:
                    continue
                for k, this in enumerate((self, other)):
                    if x in this.scope:
                        bms[k] |= 1 << (len(this.scope) - this._idx[x] - 1)
            # Now fill in 𝜎(i)
            factor._entries[i] = self._entries[bms[0]] * other._entries[bms[1]]
        return factor

    def marginalize(self, var: int) -> "Factor":
        idx = self._idx[var]
        scope = {*self.scope} - {var}
        cnt = len(scope)
        # Temporarily set the values to 0
        factor = Factor(scope, [0.0] * (1 << cnt))

        # Set the values for all indices; accesses an "unsafe" _entries
        for bm_hd in range(1 << idx):
            for bm_tl in range(1 << (cnt - idx)):
                bm_self = bm_hd << (cnt - idx + 1) | bm_tl
                bm = bm_hd << (cnt - idx) | bm_tl
                factor._entries[bm] = (
                    self._entries[bm_self] + self._entries[bm_self | (1 << (cnt - idx))]
                )
        return factor

    def reduce(self, var: int, val: Bin) -> "Factor":
        idx = self._idx[var]
        scope = {*self.scope} - {var}
        cnt = len(scope)
        # Temporarily set the values to 0
        factor = Factor(scope, [0.0] * (1 << cnt))

        # Set the values for all indices; accesses an "unsafe" _entries
        for bm_hd in range(1 << idx):
            for bm_tl in range(1 << (cnt - idx)):
                bm_self = bm_hd << (cnt - idx + 1) | bm_tl
                bm = bm_hd << (cnt - idx) | bm_tl
                factor._entries[bm] = (
                    self._entries[bm_self | (1 << (cnt - idx))]
                    if val
                    else self._entries[bm_self]
                )
        return factor

    @staticmethod
    def sum_product_eliminate_var(factors: list["Factor"], var: int) -> list["Factor"]:
        """The Sum-Product-Eliminate-Var algorithm.
        Eliminates a variable from the factors."""
        print(f"[SUM-PRODUCT-ELIMINATE-VAR] Called with {factors=}, {var=}")
        factors_used, factors_not_used = [], []
        for f in factors:
            if var in f.scope:
                factors_used.append(f)
            else:
                factors_not_used.append(f)

        if not factors_used:
            print(
                f"[SUM-PRODUCT-ELIMINATE-VAR] Returning {factors_not_used} (no factors used)"
            )
            return factors_not_used

        prod: Factor = reduce(operator.mul, factors_used)
        factor = prod.marginalize(var)
        factors_not_used.append(factor)
        print(
            f"[SUM-PRODUCT-ELIMINATE-VAR] Returning {factors_not_used} (added {factor=})"
        )
        return factors_not_used

    @staticmethod
    def sum_product_ve(factors: list["Factor"], vars: list[int]) -> "Factor":
        "The Sum-Product-VE algorithm."
        print(f"[SUM-PRODUCT-VE] Called with {factors=}, {vars=}")
        for x in vars:
            factors = Factor.sum_product_eliminate_var(factors, x)

        factor = reduce(operator.mul, factors)
        print(f"[SUM-PRODUCT-VE] Returning {factor}")
        return factor

    @staticmethod
    def cond_prob_ve(
        factors: list["Factor"],
        vars: Iterable[int],
        query_vars: Iterable[int],
        evidences: dict[int, Bin],
    ) -> tuple[float, "Factor"]:
        "The Cond-Prob-VE algorithm."
        print(
            f"[COND-PROB-VE] Called with {factors=}, {vars=}, {query_vars=}, {evidences=}"
        )
        for i in range(len(factors)):
            for x, v in evidences.items():
                if x in factors[i].scope:
                    factors[i] = factors[i].reduce(x, v)

        factor = Factor.sum_product_ve(
            factors, list({*vars} - {*query_vars} - evidences.keys())
        )
        alpha = sum(factor._entries)
        print(f"[COND-PROB-VE] Returning {alpha=}, {factor=}")
        return (alpha, factor)


class Query:
    def __init__(self, x: int, evidences: dict[int, Bin] | None = None):
        self.var = x
        self.evidences = evidences if evidences else {}

    def __repr__(self) -> str:
        return f"Query(x={self.var}, e={self.evidences})"

    def add_evidence(self, x: int, v: Bin) -> None:
        "Adds an evidence to the query."
        self.evidences[x] = v


def one_indexify(lst):
    "Convert a list into a 1-indexed dictionary. Useful for logging."
    return {i + 1: p for i, p in enumerate(lst)}


class Graph:
    "Graph represents a Bayesian network. Nodes are 1-indexed."

    def __init__(self, size: int):
        self.size = size
        self.parents: list[Sequence[int] | None] = [None] * size
        self.factors: list[Factor | None] = [None] * size

    def __repr__(self) -> str:
        return f"Graph({self.size}; parents={one_indexify(self.parents)}, factors={one_indexify(self.factors)})"

    def add_cpt(self, node: int, parents: Sequence[int], cpt: Sequence[float]) -> None:
        "Adds a CPT with given parents to the graph."
        self.factors[node - 1] = Factor.from_cpt(node, parents, cpt)
        self.parents[node - 1] = sorted(parents)

    def ask(self, query: Query) -> float:
        "Computes the probability of the query in the context of the graph."
        alpha, factor = Factor.cond_prob_ve(
            cast(list[Factor], list(self.factors)),
            range(1, self.size + 1),
            (query.var,),
            query.evidences,
        )
        return factor[1] / alpha


def parse_input(filename: str) -> tuple[Graph, Query]:
    "Parses the input file and return a Graph and a Query."
    with open(filename, "r") as f:
        spec = f.readlines()

    n = int(spec[0])
    g = Graph(n)
    for i in range(1, 2 * n + 1, 2):
        line = spec[i].split()
        parents_cnt = int(line[0])
        parents = [int(line[j]) for j in range(1, parents_cnt + 1)]
        cpts = list(map(float, spec[i + 1].split()))
        g.add_cpt(i // 2 + 1, parents, cpts)

    x, e = map(int, spec[2 * n + 1].split())
    q = Query(x)
    for i in range(-e, 0):
        idx, val = map(int, spec[i].split())
        assert val == 0 or val == 1
        q.add_evidence(idx, val)

    return g, q


def write_output(prob: float, filename: str) -> None:
    "Writes the probability to the output file."
    with open(filename, "w") as f:
        f.write(f"{prob:.3f}\n")


g, q = parse_input("in.txt")
# write_output(g.ask(q), "out.txt")


for tc in range(1, 8):
    g, q = parse_input(f"tests/test{tc}.txt")
    old_print(g.ask(q))
