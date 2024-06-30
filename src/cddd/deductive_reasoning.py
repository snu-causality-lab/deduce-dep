def deduce_dep(data, X, Y, Z, k, alpha, add_ci_set, sepsets, consets=None, ci_tester=None):
    '''
    recursively deduce a dependence statement from CIT results
    Args:
        data: given dataset
        X: variable of interest from CI query (X;Y|Z)
        Y: variable of interest from CI query (X;Y|Z)
        Z: conditioning set from CI query (X;Y|Z)
        k: recursion threshold for deduce-dep
        alpha: the significance level for CIT to use
        add_ci_set: indicator for counting the total number of CIT performed
        sepsets: a dictionary of CI queries with independence results
        consets: a dictionary of CI queries with dependence results
        ci_tester: CIT to use

    Returns: whether dependence is deducible or not (Boolean)

    '''
    if consets is None:
        consets = dict()
    if len(Z) > k:
        for z in Z:
            remaining_Z = tuple(list((set(Z) - {z})))

            for A, B, C in [(X, Y, remaining_Z), (X, z, remaining_Z), (Y, z, remaining_Z)]:
                is_already_identified = False

                if tuple(sorted([A, B])) in sepsets and sepsets[tuple(sorted([A, B]))] == C:
                    pval = 1
                    is_already_identified = True

                elif tuple(sorted([A, B])) in consets and consets[tuple(sorted([A, B]))] == C:
                    pval = 0

                else:
                    pval, _ = ci_tester.ci_test(data, A, B, C)
                    add_ci_set.append('1')

                if pval > alpha:
                    if not is_already_identified:
                        if not deduce_dep(data, A, B, C, k, alpha, add_ci_set, sepsets, consets, ci_tester=ci_tester):
                            sepsets[tuple(sorted([A, B]))] = C
                        else:
                            consets[tuple(sorted([A, B]))] = C

                else:
                    consets[tuple(sorted([A, B]))] = C

                if tuple(sorted([X, Y])) not in sepsets:
                    if (tuple(sorted([X, z])) in sepsets and sepsets[(tuple(sorted([X, z])))] == remaining_Z) or (tuple(sorted([Y, z])) in sepsets and sepsets[(tuple(sorted([Y, z])))] == remaining_Z):
                        return True

            if tuple(sorted([X, Y])) in sepsets and sepsets[tuple(sorted([X, Y]))] == remaining_Z:
                if (tuple(sorted([X, z])) not in sepsets) and (tuple(sorted([Y, z])) not in sepsets):
                    return True
    return False
