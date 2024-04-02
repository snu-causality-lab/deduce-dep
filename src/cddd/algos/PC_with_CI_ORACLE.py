from itertools import combinations

import networkx as nx


def pc_oracle(true_adj_mat, true_graph):
    num_of_variables = len(true_adj_mat)
    adj_mat = [[1 if i != j else 0 for j in range(num_of_variables)] for i in range(num_of_variables)]
    sepsets = dict()

    for k in range(num_of_variables - 1):
        for target in range(num_of_variables):
            adj_target = [i for i in range(num_of_variables) if ((tuple(sorted([target, i])) not in sepsets) and (i != target))]
            for candidate in adj_target:
                conditioning_set_pool = list(set(adj_target) - {candidate})
                if len(conditioning_set_pool) >= k:
                    k_length_conditioning_sets = combinations(conditioning_set_pool, k)
                    for cond_set in k_length_conditioning_sets:
                        ci_oracle = nx.d_separated(true_graph, {target}, {candidate}, set(cond_set))
                        if ci_oracle:
                            sepsets[tuple(sorted([target, candidate]))] = cond_set
                            adj_mat[target][candidate] = 0
                            adj_mat[candidate][target] = 0
                            adj_target.remove(candidate)
                            break
    return adj_mat
