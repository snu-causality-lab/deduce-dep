import networkx as nx
import numpy as np
import pandas as pd
import pytest

from cddd import evaluation
from cddd.algos.HITON_PC_with_CI_ORACLE import HITON_PC_oracle


class MarginalTester:
    """Supply deliberately wrong marginal decisions and fixed association scores."""

    def __init__(self, pvalue, scores=None):
        self.pvalue = pvalue
        self.scores = scores
        self.calls = []

    def ci_test(self, data, x, y, z):
        self.calls.append((x, y, tuple(z)))
        association = self.scores[y] if self.scores is not None else data.shape[1] - y
        return self.pvalue, association


@pytest.mark.parametrize("pvalue", [0.0, 1.0])
def test_oracle_uses_graph_for_marginal_inclusion_and_exclusion(pvalue):
    # In a chain plus an isolated node, only 1 is adjacent to target 0.
    graph = nx.DiGraph([(0, 1), (1, 2)])
    graph.add_node(3)
    data = pd.DataFrame([[0] * 4, [1] * 4])
    assoc = [[0] * 4 for _ in range(4)]
    tester = MarginalTester(pvalue)

    result = HITON_PC_oracle(data, assoc, 0, 0.05, graph, ci_tester=tester)

    assert set(result) == {1}
    # Statistical associations still determine candidate ordering, not CI.
    assert assoc[0] == [0, 3, 2, 1]
    assert tester.calls == [(0, 1, ()), (0, 2, ()), (0, 3, ())]


@pytest.mark.parametrize("order,expected", [
    ([0, 1, 3, 4], {0, 3, 4}),
    ([1, 3, 4, 0], {0, 3}),
])
def test_oracle_preserves_nonsymmetric_algorithm_reference(order, expected):
    # The reference remains NonSym, whose result can depend on the order.
    # Target 2's exact graph neighbors are {0, 3}; 4 can be retained too.
    graph = nx.DiGraph([(0, 1), (0, 2), (1, 3), (1, 4), (2, 3), (3, 4)])
    data = pd.DataFrame([[0] * 5, [1] * 5])
    assoc = [[0] * 5 for _ in range(5)]
    scores = {node: 5 - index for index, node in enumerate(order)}

    result = HITON_PC_oracle(data, assoc, 2, 0.05, graph,
                             ci_tester=MarginalTester(1.0, scores))

    assert set(result) == expected


def write_inputs(tmp_path, adjacency, repetitions):
    graph_path = tmp_path / "graph.txt"
    np.savetxt(graph_path, adjacency, fmt="%d")
    prefix = str(tmp_path / "data")
    data = pd.DataFrame([[0] * len(adjacency), [1] * len(adjacency)])
    for index in range(1, repetitions + 1):
        data.to_csv(prefix + str(index) + ".csv", index=False)
    return prefix, str(graph_path)


@pytest.mark.parametrize("has_edge,pvalue", [(True, 1.0), (False, 0.0)])
def test_evaluation_does_not_share_statistical_errors_with_reference(
        tmp_path, has_edge, pvalue):
    # These two-node references are known independently: {1}, {0} for
    # the edge, and two empty sets otherwise. The statistical learner errs.
    adjacency = [[0, int(has_edge)], [0, 0]]
    prefix, graph_path = write_inputs(tmp_path, adjacency, repetitions=1)

    result = evaluation.HITON_PC_evaluation(
        prefix, 2, [0, 1], graph_path, "NO", filenumber=1,
        reliability_criterion="no", ci_tester=MarginalTester(pvalue))

    assert result[:3] == (0.0, 0.0, 0.0)
    assert result[5:8] == (0.0, 0.0, 0.0)


def test_score_std_uses_each_repetition_mean(tmp_path, monkeypatch):
    adjacency = [[0, 1, 0], [0, 0, 1], [0, 0, 0]]
    prefix, graph_path = write_inputs(tmp_path, adjacency, repetitions=2)
    # Repeat 1 recovers the chain. Repeat 2 has an extra edge at target 0,
    # misses one neighbor at target 1, and misses target 2's only neighbor.
    learned = iter([[1], [0, 2], [1], [1, 2], [0], []])

    def controlled_hiton(*args, **kwargs):
        return next(learned), {}, 4

    monkeypatch.setattr(evaluation, "HITON_PC", controlled_hiton)
    result = evaluation.HITON_PC_evaluation(
        prefix, 3, [0, 1, 2], graph_path, "NO", filenumber=2,
        ci_tester=MarginalTester(0.0))

    # Repetition means: F1 = [1, 4/9], precision = recall = [1, 1/2].
    assert result[:3] == pytest.approx((13 / 18, 0.75, 0.75))
    assert result[5:8] == pytest.approx((5 / 18, 0.25, 0.25))
    assert result[3] == 4
    assert result[8] == 0
