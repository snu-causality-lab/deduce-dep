from itertools import combinations

import numpy as np

from cddd.cit import cond_indep_test


def pc_simple(data, target, alpha, is_discrete=True):
    size_of_dataset, num_of_variables = np.shape(data)
    sepset = [[] for _ in range(num_of_variables)]
    ci_number = 0
    k = 0

    PC = [i for i in range(num_of_variables) if i != target]

    while len(PC) > k:
        PC_temp = PC.copy()
        for x in PC_temp:
            condition_subsets = [i for i in PC_temp if i != x]
            if len(condition_subsets) >= k:
                css = combinations(condition_subsets, k)
                for cond_set in css:
                    pval, _ = cond_indep_test(data, target, x, cond_set, is_discrete)
                    ci_number += 1
                    if pval > alpha:
                        sepset[x] = cond_set
                        PC.remove(x)
                        break
        k += 1

    return PC, sepset, ci_number
