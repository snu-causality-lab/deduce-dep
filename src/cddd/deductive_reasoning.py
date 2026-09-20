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
        sepsets: maps variable pairs to a cached independent conditioning set
        consets: maps variable pairs to a cached dependent conditioning set
        ci_tester: CIT to use

    Returns: whether dependence is deducible or not (Boolean)

    '''
    if consets is None:
        consets = dict()
    if len(Z) > k:
        for z in Z:
            remaining_Z = tuple(sorted(set(Z) - {z}))
            independent = []

            for A, B, C in [(X, Y, remaining_Z), (X, z, remaining_Z), (Y, z, remaining_Z)]:
                pair = tuple(sorted([A, B]))
                if pair in sepsets and tuple(sorted(sepsets[pair])) == C:
                    is_independent = True
                elif pair in consets and tuple(sorted(consets[pair])) == C:
                    is_independent = False
                else:
                    pval, _ = ci_tester.ci_test(data, A, B, C)
                    add_ci_set.append('1')
                    is_independent = pval > alpha
                    if is_independent:
                        is_independent = not deduce_dep(
                            data, A, B, C, k, alpha, add_ci_set, sepsets,
                            consets, ci_tester=ci_tester)
                    if is_independent:
                        sepsets[pair] = C
                    else:
                        consets[pair] = C

                # Each cache retains only one conditioning set per pair.
                # Apply the rules to these actual query results, never to
                # pair membership or absence under another conditioning set.
                independent.append(is_independent)
                if len(independent) > 1 and not independent[0] and is_independent:
                    return True

            if independent[0] and not independent[1] and not independent[2]:
                return True
    return False
