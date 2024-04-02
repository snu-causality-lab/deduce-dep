from itertools import combinations

import networkx as nx


def pc_stable_oracle(true_adj_mat, true_graph):
    num_of_variables = len(true_adj_mat)
    adj_mat = [[1 if i != j else 0 for j in range(num_of_variables)] for i in range(num_of_variables)]
    sepsets = dict()

    for k in range(num_of_variables - 1):
        marker = []
        for target in range(num_of_variables):
            adj_target = [i for i in range(num_of_variables) if ((tuple(sorted([target, i])) not in sepsets) and (i != target))]
            for candidate in adj_target:
                conditioning_set_pool = list(set(adj_target) - {candidate})
                if len(conditioning_set_pool) >= k:
                    k_length_conditioning_sets = combinations(conditioning_set_pool, k)
                    for cond_set in k_length_conditioning_sets:
                        ci_oracle = nx.d_separated(true_graph, {target}, {candidate}, set(cond_set))
                        if ci_oracle:
                            marker.append([tuple(sorted([target, candidate])), cond_set])
                            break

        for pair_tuple, cond_set in marker:
            sepsets[pair_tuple] = cond_set
            var1, var2 = pair_tuple
            adj_mat[var1][var2] = 0
            adj_mat[var2][var1] = 0

    return adj_mat
