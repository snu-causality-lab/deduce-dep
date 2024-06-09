from itertools import combinations

import networkx as nx
import numpy as np

from cddd.algos.algo_utils import estimate_cpdag
from cddd.utils import DAG_to_CPDAG

def pc_stable_oracle(true_adj_mat, true_graph, is_orientation=False):
    if not is_orientation:
        true_skeleton_mat = np.array(true_adj_mat)
        true_skeleton_mat = true_skeleton_mat + true_skeleton_mat.T
        return true_skeleton_mat

    num_of_variables = len(true_adj_mat)
    adj_mat = [[1 if i != j else 0 for j in range(num_of_variables)] for i in range(num_of_variables)]
    sepsets = dict()

    for k in range(num_of_variables - 1):
        marker = []
        for target in range(num_of_variables):
            adj_target = [i for i in range(num_of_variables) if (adj_mat[i][target] == 1) and (i != target)]
            for candidate in adj_target:
                conditioning_set_pool = list(set(adj_target) - {candidate})
                if len(conditioning_set_pool) >= k:
                    k_length_conditioning_sets = combinations(conditioning_set_pool, k)
                    for cond_set in k_length_conditioning_sets:
                        ci_oracle = nx.d_separated(true_graph, {target}, {candidate}, set(cond_set))
                        if ci_oracle:
                            sepsets[tuple(sorted([target, candidate]))] = cond_set
                            marker.append([tuple(sorted([target, candidate])), cond_set])
                            break

        for pair_tuple, cond_set in marker:
            sepsets[pair_tuple] = cond_set
            var1, var2 = pair_tuple
            adj_mat[var1][var2] = 0
            adj_mat[var2][var1] = 0

    adj_mat = np.array(adj_mat)
    skel_graph = nx.DiGraph(adj_mat)
    dag = estimate_cpdag(skel_graph, sepsets)
    adj_mat = nx.to_numpy_array(dag)

    return adj_mat


