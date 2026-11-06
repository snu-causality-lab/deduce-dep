from itertools import combinations

import pytest

from cddd.deductive_reasoning import deduce_dep


def d_separated(n, edges, x, y, conditioning):
    """Independent oracle: separation in the ancestral moral graph."""
    parents = {v: set() for v in range(n)}
    for parent, child in edges:
        parents[child].add(parent)
    ancestors = {x, y} | set(conditioning)
    pending = list(ancestors)
    while pending:
        for parent in parents[pending.pop()]:
            if parent not in ancestors:
                ancestors.add(parent)
                pending.append(parent)
    adjacency = {v: set() for v in ancestors}
    for child in ancestors:
        for parent in parents[child]:
            adjacency[parent].add(child)
            adjacency[child].add(parent)
        for a, b in combinations(parents[child], 2):
            adjacency[a].add(b)
            adjacency[b].add(a)
    seen = set(conditioning) | {x}
    pending = [x]
    while pending:
        for neighbor in adjacency[pending.pop()] - seen:
            if neighbor == y:
                return False
            seen.add(neighbor)
            pending.append(neighbor)
    return True


class OracleTester:
    def __init__(self, n, edges):
        self.n = n
        self.edges = edges
        self.calls = []

    def ci_test(self, data, x, y, conditioning):
        self.calls.append((x, y, tuple(conditioning)))
        return float(d_separated(self.n, self.edges, x, y, conditioning)), 0


class NoCallsTester:
    def ci_test(self, *args):
        pytest.fail("A cached or below-threshold query must not run a CI test")


COLLIDER_EDGES = [(0, 2), (1, 2), (2, 3)]


@pytest.mark.parametrize("initial_sepsets", [{}, {(0, 1): ()}])
@pytest.mark.parametrize("conditioning", [(2, 3), (3, 2)])
def test_deduction_is_not_blocked_by_another_conditioning_set(
        initial_sepsets, conditioning):
    # X -> Z <- Y and Z -> W: X and Y are marginally independent,
    # but dependent given Z,W. Both cache states must deduce dependence.
    tester = OracleTester(4, COLLIDER_EDGES)
    count = []
    assert deduce_dep(None, 0, 1, conditioning, 1, .05, count,
                      dict(initial_sepsets), {}, tester)
    assert len(count) == len(tester.calls)


def test_warm_cache_from_an_earlier_valid_query_does_not_block_deduction():
    tester = OracleTester(6, COLLIDER_EDGES)
    sepsets, consets, count = {}, {}, []
    # Nodes 4 and 5 are isolated. This call legitimately caches 0 _||_ 1 | 4.
    assert not deduce_dep(None, 0, 5, (1, 4), 1, .05, count,
                          sepsets, consets, tester)
    assert sepsets[(0, 1)] == (4,)
    assert deduce_dep(None, 0, 1, (2, 3), 1, .05, count,
                      sepsets, consets, tester)
    assert len(count) == len(tester.calls)


@pytest.mark.parametrize("empty", [[], ()])
def test_empty_list_and_tuple_cache_entries_are_equivalent(empty):
    # 0 -> 2 <- 1: all three marginal results are already cached.
    sepsets = {(0, 1): empty}
    consets = {(0, 2): empty, (1, 2): empty}
    count = []
    assert deduce_dep(None, 0, 1, (2,), 0, .05, count,
                      sepsets, consets, NoCallsTester())
    assert count == []


def test_conditioning_set_order_does_not_change_cache_identity():
    sepsets = {(0, 1): [4, 3]}
    consets = {(0, 2): [4, 3], (1, 2): (3, 4)}
    assert deduce_dep(None, 0, 1, (2, 3, 4), 2, .05, [],
                      sepsets, consets, NoCallsTester())


def test_deduction_can_stop_after_two_sufficient_cached_results():
    # 0 _!||_ 1 and 0 _||_ 2 imply 0 _!||_ 1 | 2.
    assert deduce_dep(None, 0, 1, (2,), 0, .05, [],
                      {(0, 2): []}, {(0, 1): []}, NoCallsTester())


@pytest.mark.parametrize("conditioning,k", [((), 0), ((2,), 1), ((2, 3), 2)])
def test_recursion_threshold_prevents_lower_order_tests(conditioning, k):
    assert not deduce_dep(None, 0, 1, conditioning, k, .05, [],
                          {}, {}, NoCallsTester())


def test_recursively_corrected_result_is_used_by_the_parent_rule():
    class NoisyTester(OracleTester):
        def ci_test(self, data, x, y, conditioning):
            pval, statistic = super().ci_test(data, x, y, conditioning)
            if {x, y} == {0, 1} and set(conditioning) == {3}:
                # A false independence is corrected using marginal results.
                pval = 1.
            return pval, statistic

    tester = NoisyTester(4, COLLIDER_EDGES)
    count = []
    assert deduce_dep(None, 0, 1, (2, 3), 0, .05, count, {}, {}, tester)
    assert (0, 1, (3,)) in tester.calls
    assert (0, 1, ()) in tester.calls
    assert len(count) == len(tester.calls)


def subsets(values):
    return [subset for size in range(len(values) + 1)
            for subset in combinations(values, size)]


def expected_deduction(n, edges, x, y, conditioning, k):
    if len(conditioning) <= k:
        return False
    for z in conditioning:
        remaining = set(conditioning) - {z}
        xy, xz, yz = [d_separated(n, edges, a, b, remaining)
                      for a, b in [(x, y), (x, z), (y, z)]]
        if (not xy and (xz or yz)) or (xy and not xz and not yz):
            return True
    return False


@pytest.mark.parametrize("k", [0, 1])
def test_all_four_node_ordered_dags_with_sound_warm_caches(k):
    # Every DAG on this fixed topological order; variable relabeling is
    # immaterial. Compare with a moral-graph oracle, including independent
    # targets so a dependence correction cannot silently be unsound.
    possible_edges = list(combinations(range(4), 2))
    for mask in range(1 << len(possible_edges)):
        edges = [edge for i, edge in enumerate(possible_edges) if mask & (1 << i)]
        sepsets, consets = {}, {}
        for x, y in combinations(range(4), 2):
            for conditioning in subsets([v for v in range(4) if v not in (x, y)]):
                cache = (sepsets if d_separated(4, edges, x, y, conditioning)
                         else consets)
                cache.setdefault((x, y), conditioning)
        tester = OracleTester(4, edges)
        for x, y in combinations(range(4), 2):
            for conditioning in subsets([v for v in range(4) if v not in (x, y)]):
                actual = deduce_dep(None, x, y, conditioning, k, .05, [],
                                    sepsets, consets, tester)
                expected = expected_deduction(4, edges, x, y, conditioning, k)
                assert actual == expected, (edges, x, y, conditioning, k)
                if actual:
                    assert not d_separated(4, edges, x, y, conditioning)
