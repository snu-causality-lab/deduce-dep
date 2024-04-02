from itertools import combinations

import networkx as nx
import numpy as np


def HITON_PC_oracle(data, assoc, target, alpha, is_discrete, true_graph, ci_tester=None):
    size_of_dataset, num_of_variables = np.shape(data)
    sepsets = dict()
    OPEN = []
    TPC = []

    total_variables = [var for var in range(num_of_variables) if var != target]
    for x in total_variables:
        # pval_gp, dep_gp = cond_indep_test(data, target, x, [], is_discrete)
        pval_gp, dep_gp = ci_tester.ci_test(data, target, x, [])
        assoc[target][x] = dep_gp

        if pval_gp <= alpha:
            OPEN.append(x)
        else:
            sepsets[tuple(sorted([target, x]))] = []

    # sorted by dep from max to min
    OPEN = sorted(OPEN, key=lambda x_: assoc[target][x_], reverse=True)

    """ sp """
    for x in OPEN:
        # NEW_VAR = x
        TPC.append(x)
        TPC_index = len(TPC)

        while TPC_index > 0:
            TPC_index -= 1
            TPC_var = TPC[TPC_index]
            remaining_TPC = [var for var in TPC if var != TPC_var]

            cond_sets = []
            for cond_set_len in range(1, len(remaining_TPC) + 1):
                cond_sets += list(combinations(remaining_TPC, cond_set_len))

            for cond_set in cond_sets:
                if (TPC_var != x) and (x not in cond_set):
                    continue

                ci_oracle = nx.d_separated(true_graph, {target}, {TPC_var}, set(cond_set))
                if ci_oracle:
                    sepsets[tuple(sorted([target, TPC_var]))] = cond_set
                    TPC.remove(TPC_var)
                    break

    return list(set(TPC))
