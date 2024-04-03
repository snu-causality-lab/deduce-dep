import numpy as np


def INTER_IAMB(data, target, alpha, ci_tester=None):
    size_of_dataset, num_of_variables = np.shape(data)
    sepset = [[] for _ in range(num_of_variables)]
    ci_number = 0
    MB = []

    circulateFlag = True
    removeSet = []

    while circulateFlag:
        circulateFlag = False
        dep_temp = - float("inf")
        pval_temp = 1
        max_s = None

        variables = [i for i in range(num_of_variables) if i != target and i not in MB and i not in removeSet]
        # growing phase
        for s in variables:
            # pval_gp, dep_gp = cond_indep_test(data, target, s, MB, is_discrete)
            pval_gp, dep_gp = ci_tester.ci_test(data, target, s, MB)
            ci_number += 1

            if dep_gp > dep_temp:
                dep_temp = dep_gp
                max_s = s
                pval_temp = pval_gp

        if pval_temp <= alpha:
            circulateFlag = True
            MB.append(max_s)

        if not circulateFlag:
            break

        # shrinking phase
        mb_index = len(MB)
        while mb_index >= 0:
            mb_index -= 1
            x = MB[mb_index]

            subsets_Variables = [i for i in MB if i != x]
            # pval_sp, dep_sp = cond_indep_test(data, target, x, subsets_Variables, is_discrete)
            pval_sp, dep_sp = ci_tester.ci_test(data, target, x, subsets_Variables)
            ci_number += 1

            if pval_sp > alpha:
                MB.remove(x)
                sepset[x] = subsets_Variables
                removeSet.append(x)

                if x == max_s:
                    break

    return list(set(MB)), sepset, ci_number
