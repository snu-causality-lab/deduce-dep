from functools import reduce
from itertools import combinations
import numpy as np
from cddd.deductive_reasoning import deduce_dep


def get_num_of_parameters(data, target, x, z):
    num_of_domain_target = len(data[target].unique())
    num_of_domain_x = len(data[x].unique())
    if z:
        levels_of_domain_z = list(map(lambda x_: len(data[x_].unique()), z))
        num_of_domain_z = reduce(lambda x_, y_: x_ * y_, levels_of_domain_z)
    else:
        num_of_domain_z = 1

    num_of_parameters = num_of_domain_x * num_of_domain_target * num_of_domain_z
    return num_of_parameters


def HITON_PC(data, assoc, target, alpha, reliability_criterion='classic', K=1, ci_tester=None):
    size_of_dataset, num_of_variables = np.shape(data)
    # Redesign sepsets
    sepsets = dict()
    consets = dict()
    add_ci_set = []

    OPEN = []
    TPC = []

    ci_number = 0

    h_ps, is_deductive_reasoning, max_k = set_reliability_criterion(reliability_criterion)

    total_variables = [var for var in range(num_of_variables) if var != target]
    for x in total_variables:
        # classic reliability criterion : h-ps
        num_of_parameters_for_cit = get_num_of_parameters(data, target, x, [])
        if size_of_dataset >= h_ps * num_of_parameters_for_cit:
            # pval_gp, dep_gp = cond_indep_test(data, target, x, [], is_discrete)
            pval_gp, dep_gp = ci_tester.ci_test(data, target, x, [])
            assoc[target][x] = dep_gp
            ci_number += 1

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

            # classic reliability criterion : max_k
            max_length_for_cond_set = min(max_k, len(remaining_TPC))

            cond_sets = []
            for cond_set_len in range(1, max_length_for_cond_set + 1):
                cond_sets += list(combinations(remaining_TPC, cond_set_len))

            for cond_set in cond_sets:
                if (TPC_var != x) and (x not in cond_set):
                    continue

                # classic reliability criterion : h-ps
                num_of_parameters_for_cit = get_num_of_parameters(data, target, TPC_var, cond_set)
                if size_of_dataset >= (h_ps * num_of_parameters_for_cit):
                    if tuple(sorted([target, TPC_var])) in sepsets and sepsets[
                        tuple(sorted([target, TPC_var]))] == cond_set:
                        pval_rm = 1
                    elif tuple(sorted([target, TPC_var])) in consets and consets[
                        tuple(sorted([target, TPC_var]))] == cond_set:
                        pval_rm = 0
                    else:
                        # pval_rm, dep_rm = cond_indep_test(data, target, TPC_var, cond_set, is_discrete)
                        pval_rm, dep_rm = ci_tester.ci_test(data, target, TPC_var, cond_set)
                        ci_number += 1

                    if pval_rm > alpha:
                        # new reliability criterion : deductive reasoning
                        if is_deductive_reasoning:
                            if not deduce_dep(data, target, TPC_var, cond_set, K, alpha, add_ci_set, sepsets, consets,
                                              ci_tester=ci_tester):
                                sepsets[tuple(sorted([target, TPC_var]))] = cond_set
                                TPC.remove(TPC_var)
                                break
                            else:
                                consets[tuple(sorted([target, TPC_var]))] = cond_set
                        else:
                            sepsets[tuple(sorted([target, TPC_var]))] = cond_set
                            TPC.remove(TPC_var)
                            break
                    else:
                        consets[tuple(sorted([target, TPC_var]))] = cond_set

    ci_number += len(add_ci_set)
    return list(set(TPC)), sepsets, ci_number


def set_reliability_criterion(reliability_criterion):
    if reliability_criterion == 'no':
        h_ps = 0
        max_k = float('inf')
        is_deductive_reasoning = False
    elif reliability_criterion == 'classic':
        h_ps = 5
        max_k = float('inf')
        is_deductive_reasoning = False
    elif reliability_criterion == 'deductive_reasoning':
        h_ps = 0
        max_k = float('inf')
        is_deductive_reasoning = True
    elif reliability_criterion == 'both':
        h_ps = 5
        max_k = float('inf')
        is_deductive_reasoning = True
    else:
        raise AssertionError(f'unknown reliability criterion: {reliability_criterion}.')

    return h_ps, is_deductive_reasoning, max_k
