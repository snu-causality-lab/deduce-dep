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


def correction(ResPC, rule):
    if rule == 'AND':
        AND_correction(ResPC)

    elif rule == 'OR':
        OR_correction(ResPC)

    elif rule == 'NO':
        pass

    else:
        raise AssertionError(f'unknown rule: {rule}')
