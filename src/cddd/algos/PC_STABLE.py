from itertools import combinations

import networkx as nx
import numpy as np

from cddd.algos.algo_utils import estimate_cpdag
from cddd.deductive_reasoning import deduce_dep


def pc_stable(data, alpha, reliability_criterion='classic', is_orientation=False, K=1, ci_tester=None):
    '''
    learn a graphical structure from sampled data with pc-stable algorithm.

    Args:
        data: the sampled dataset
        alpha: the significance level for CIT to use
        reliability_criterion: the reliability criterion for structure learning
        is_orientation: whether an orientation step is performed
        K: threshold value for deduce-dep
        ci_tester: CIT to use

    Returns:
        adj_mat: adjacency matrix from structure learning
        sepsets: a dictionary for CI queries with independence results
        ci_number: the total number of CIT performed

    '''
    size_of_dataset, num_of_variables = np.shape(data)
    adj_mat = [[1 if i != j else 0 for j in range(num_of_variables)] for i in range(num_of_variables)]
    sepsets = dict()
    consets = dict()
    sepsets_for_orientation = dict()

    add_ci_set = []
    ci_number = 0
    is_deductive_reasoning = (reliability_criterion == 'deductive_reasoning')

    for k in range(num_of_variables - 1):
        marker = []
        for target in range(num_of_variables):
            adj_target = [i for i in range(num_of_variables) if (adj_mat[i][target] == 1) and (i != target)]
            for candidate in adj_target:
                conditioning_set_pool = list(set(adj_target) - {candidate})
                if len(conditioning_set_pool) >= k:
                    k_length_conditioning_sets = combinations(conditioning_set_pool, k)
                    for cond_set in k_length_conditioning_sets:
                        pval, _ = ci_tester.ci_test(data, target, candidate, cond_set)
                        ci_number += 1
                        if pval > alpha:
                            if is_deductive_reasoning:
                                if not deduce_dep(data, target, candidate, cond_set, K, alpha, add_ci_set, sepsets, consets, ci_tester=ci_tester):
                                    sepsets[tuple(sorted([target, candidate]))] = cond_set
                                    sepsets_for_orientation[tuple(sorted([target, candidate]))] = cond_set
                                    marker.append([tuple(sorted([target, candidate])), cond_set])
                                    break
                                else:
                                    consets[tuple(sorted([target, candidate]))] = cond_set

                            else:
                                sepsets[tuple(sorted([target, candidate]))] = cond_set
                                sepsets_for_orientation[tuple(sorted([target, candidate]))] = cond_set
                                marker.append([tuple(sorted([target, candidate])), cond_set])
                                break
                        else:
                            consets[tuple(sorted([target, candidate]))] = cond_set

        for pair_tuple, cond_set in marker:
            sepsets[pair_tuple] = cond_set
            var1, var2 = pair_tuple
            adj_mat[var1][var2] = 0
            adj_mat[var2][var1] = 0

    ci_number += len(add_ci_set)
    if not is_orientation:
        adj_mat = np.array(adj_mat)
        return adj_mat, sepsets, ci_number
    else:
        adj_mat = np.array(adj_mat)
        skel_graph = nx.DiGraph(adj_mat)
        dag = estimate_cpdag(skel_graph, sepsets_for_orientation)
        adj_mat = nx.to_numpy_array(dag)
        return adj_mat, sepsets, ci_number
