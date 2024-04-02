from cddd.cit import cond_indep_test


def deduce_dep(sepsets, ResPC, data, alpha=0.01):
    marker = [[0] * len(ResPC) for _ in range(len(ResPC))]  # for order-independent correction

    for target in range(len(ResPC)):
        for var in range(len(ResPC)):
            if len(sepsets[target][var]) > 1:
                Z = sepsets[target][var]
                for z in Z:
                    remaining_Z = [elem for elem in Z if elem != z]

                    pval, _ = cond_indep_test(data, target, z, remaining_Z)

                    if pval > alpha:
                        sepsets[target][var] = []
                        marker[target][var] = 1
                        break

    for target in range(len(ResPC)):
        for var in range(len(ResPC)):
            if marker[target][var] == 1:
                ResPC[target].append(var)


def AND_correction(ResPC):
    for target in range(len(ResPC)):
        for var in ResPC[target]:
            if target not in ResPC[var]:
                ResPC[target].remove(var)


def OR_correction(ResPC):
    for target in range(len(ResPC)):
        for var in ResPC[target]:
            if target not in ResPC[var]:
                ResPC[var].append(target)


def correction(ResPC, rule, data, sepsets, alpha):
    if rule == 'AND':
        AND_correction(ResPC)

    elif rule == 'OR':
        OR_correction(ResPC)

    elif rule == 'DEDUCE':
        deduce_dep(sepsets, ResPC, data, alpha)

    elif rule == 'NO':
        pass

    else:
        raise AssertionError(f'unknown rule: {rule}')
