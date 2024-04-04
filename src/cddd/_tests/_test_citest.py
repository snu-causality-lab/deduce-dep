import numpy as np
import pandas as pd

from cddd.cit import PartialCorrelation

if __name__ == '__main__':
    z = np.random.randn(100)
    x = z + np.random.randn(100)
    y = z + x + np.random.randn(100)
    parcorr = PartialCorrelation()

    xyz = pd.DataFrame(np.vstack([x, y, z]).T)
    xyz.columns = ('X', 'Y', 'Z')
    print(xyz.columns)
    pval, dep = parcorr.ci_test(xyz, 'X', 'Y', {'Z'})
    print(pval, dep)

    xyz = pd.DataFrame(np.vstack([x, y, z]).T)
    xyz.columns = (0, 1, 2)
    print(xyz.columns)
    pval, dep = parcorr.ci_test(xyz, 0, 1, {2})
    print(pval, dep)
