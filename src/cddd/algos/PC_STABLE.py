from itertools import combinations

import numpy as np

from cddd.cit import cond_indep_test
from cddd.deductive_reasoning import deduce_dep


def pc_stable(data, alpha, is_discrete=True, reliability_criterion='classic', K=1):
    size_of_dataset, num_of_variables = np.shape(data)
    adj_mat = [[1 if i != j else 0 for j in range(num_of_variables)] for i in range(num_of_variables)]
    sepsets = dict()
    consets = dict()

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
                        pval, _ = cond_indep_test(data, target, candidate, cond_set, is_discrete)
                        ci_number += 1
                        if pval > alpha:
                            if is_deductive_reasoning:
                                if not deduce_dep(data, target, candidate, cond_set, K, alpha, add_ci_set, sepsets, consets):
                                    sepsets[tuple(sorted([target, candidate]))] = cond_set
                                    marker.append([tuple(sorted([target, candidate])), cond_set])
                                    break
                                else:
                                    consets[tuple(sorted([target, candidate]))] = cond_set

                            else:
                                sepsets[tuple(sorted([target, candidate]))] = cond_set
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
    return adj_mat, sepsets, ci_number
