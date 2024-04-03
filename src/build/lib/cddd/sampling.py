import random
from collections import deque
from functools import lru_cache
from itertools import product, combinations
from typing import AbstractSet, Dict, Tuple, Sequence, Collection, List, TypeVar, Set, Hashable, Generic, Iterable

import networkx as nx
import numpy as np
import pandas as pd
from numpy.random import randint, choice, rand
from scipy.stats import chi2_contingency

CPTType = Dict[Tuple[int, ...], Sequence[float]]
T = TypeVar('T')
H = TypeVar('H', bound=Hashable)


def shuffled(xs: Iterable[T]) -> List[T]:
    xs = list(xs)
    random.shuffle(xs)
    return xs


def unions(xss: Collection[AbstractSet[T]]) -> Set[T]:
    out = set()
    for xs in xss:
        out |= xs
    return out


def logical_and(xs):
    iter_xs = iter(xs)
    out = next(iter_xs)
    for x in iter_xs:
        out &= x
    return out


def bootstrap(data, B=500):
    # input : original data, B
    # output : B개의 bootstrapped sub-dataset
    bootstrapped_datasets = [data.sample(frac=1, replace=True) for _ in range(B)]
    return bootstrapped_datasets


class visited_queue(Generic[H]):
    def __init__(self, xs: Collection[H]):
        self.queue = deque(xs)
        self.visited = set(xs)

    def extend(self, xs: Collection[H]):
        xs = set(xs) - self.visited
        self.visited |= xs
        for x in xs:
            self.queue.append(x)

    def pop(self) -> H:
        return self.queue.pop()

    def __bool__(self):
        return bool(self.queue)


# assume
class BayesianNetwork:
    def __init__(self, G: nx.DiGraph, CPTs: Dict[int, CPTType]):
        self.G = G
        self.CPTs = CPTs

        self.Vs = frozenset(self.G.nodes)
        self.n_variables = len(self.Vs)
        self.variable_order = tuple(nx.topological_sort(G))  # compute order of variables

    # perform prior sampling
    def sample(self, n_instances=10000) -> pd.DataFrame:
        # perform prior sampling
        instances = np.zeros((n_instances, self.n_variables))

        for i in range(n_instances):
            instance = instances[i]
            for v_i in self.variable_order:
                pa_i = tuple(instance[_] for _ in sorted(self.Pa(v_i)))  # retrieve parents' values to be used in `prob'
                xxx = choice(2, 1, p=self.CPTs[v_i][pa_i])
                instance[v_i] = xxx[0]  # use self.prob to sample a value of the variable.

        return pd.DataFrame(instances, columns=list(range(self.n_variables)))

    def is_d_separated(self,
                       Xs: AbstractSet[int],
                       Ys: AbstractSet[int],
                       Zs: AbstractSet[int] = frozenset()) -> bool:
        # test xs _||_ ys | zs
        Xs, Ys = Xs - Zs, Ys - Zs
        if Xs & Ys:
            return False
        return all(self.__is_d_separated(x, y, Zs) for x, y in product(Xs, Ys))

    @lru_cache
    def Pa(self, u):
        return frozenset(self.G.predecessors(u))

    @lru_cache
    def Ch(self, u):
        return frozenset(self.G.successors(u))

    @lru_cache
    def An(self, x):
        return frozenset({x} |
                         self.Pa(x) |
                         unions([self.An(z) for z in self.Pa(x)]))

    @lru_cache
    def De(self, x):
        return frozenset({x} |
                         self.Ch(x) |
                         unions([self.De(z) for z in self.Ch(x)]))

    def __is_d_separated(self,
                         X: int,
                         Y: int,
                         Zs: AbstractSet[int] = frozenset()) -> bool:
        # test x _||_ y | zs
        assert X not in Zs
        assert Y not in Zs
        assert X != Y

        queue = visited_queue([('>', X), ('<', X)])
        while queue:
            arrow_head, at = queue.pop()
            if at == Y:
                return False
            if arrow_head == '>':
                if at not in Zs:  # -> ->
                    queue.extend([('>', _) for _ in self.Ch(at)])
                if self.De(at) & Zs:  # -> <- only if the collider is active
                    queue.extend([('<', _) for _ in self.Pa(at)])

            elif at not in Zs:  # <- ->, or <- <-
                queue.extend([('>', _) for _ in self.Ch(at)])
                queue.extend([('<', _) for _ in self.Pa(at)])

        return True


# simply, perform chi-square test for each condition
def check_conditional_independence(data: pd.DataFrame, X, Y, Zs: AbstractSet[int] = frozenset(), alpha=0.05) -> bool:
    # xs, ys, zs be sets of variables (column indices) to check xs _||_ ys | zs
    for zs in product([0, 1], repeat=len(Zs)):
        condition = logical_and([data[Z] == z for Z, z in zip(Zs, zs)])
        selected = data[condition] if Zs else data
        _, p, *_ = chi2_contingency(pd.crosstab(selected[X], selected[Y]))
        if p < alpha:
            return False

    return True


def random_DAG(num_vars: int, num_edges: int) -> nx.DiGraph:
    # randomly generate an acyclic graph
    order = shuffled(range(num_vars))
    edge_candidates = combinations(order, 2)
    edges = [(x, y) for x, y in shuffled(edge_candidates)[:num_edges]]
    G = nx.DiGraph(edges)
    G.add_nodes_from(range(num_vars))  # ensure nodes without edges are in the graph
    return G


def Erdos_Renyi_DAG(num_vars: int, num_edges: int) -> nx.DiGraph:
    G = nx.gnm_random_graph(num_vars, num_edges)
    adjmat_G = nx.to_numpy_array(G)
    adjmat_DAG = np.tril(adjmat_G, -1)
    DAG = nx.DiGraph(adjmat_DAG)
    return DAG


def random_CPTs(G: nx.DiGraph) -> Dict[int, CPTType]:
    # randomly generate a CPT for every variable
    cpts = dict()
    for v_i in G.nodes:
        cpt_i = dict()
        parents = tuple(sorted(G.predecessors(v_i)))
        for pa_i in product([0, 1], repeat=len(parents)):  # for all parents values
            pr = rand() * 0.3 + 0.1  # 0.1~0.4
            if randint(2):  # 0 or 1
                pr = 1 - pr
            cpt_i[pa_i] = [pr, 1 - pr]
        cpts[v_i] = cpt_i
    return cpts


def random_BN(num_vars, num_edges) -> BayesianNetwork:
    graph = random_DAG(num_vars, num_edges)
    CPTs = random_CPTs(graph)
    return BayesianNetwork(graph, CPTs)
