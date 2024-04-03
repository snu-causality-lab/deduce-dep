# if __name__ == '__main__':
#     import statsmodels.formula.api as sm
#
#     df = pd.read_csv('/Users/sanghacklee/Dropbox/python_projs/CD_DD/data/Sampled_datasets/ER_10_12_1000_v30.csv')
#     df = df.rename(columns={str(_): chr(ord('A') + _)+'_a0' for _ in range(10)})
#     X = 'A_a0'
#     Y = 'B_a0'
#     Z = ('C_a0', 'D_a0', 'E_a0')
#     result = sm.ols(formula="A_a0 ~ B_a0 + C_a0 + D_a0 + E_a0", data=df).fit()
#     p_values = result.summary2().tables[1]['P>|t|']
#     print(p_values['B_a0'])

import numpy as np
import pandas as pd
from causallearn.utils.cit import CIT

from cddd.cit import PartialCorrelation, KernelCITest

if __name__ == '__main__':
    z = np.random.randn(100)
    x = z + np.random.randn(100)
    y = z - 0.4 * x + np.random.randn(100)
    parcorr = PartialCorrelation()

    xyz = pd.DataFrame(np.vstack([x, y, z]).T)
    xyz.columns = ('X', 'Y', 'Z')
    print(xyz.columns)
    pval, dep = parcorr.ci_test(xyz, 'X', 'Y', {'Z'})
    print(pval, dep)

    kci = KernelCITest()
    pval, dep = kci.ci_test(xyz, 'X','Y',{'Z'})
    print(pval, dep)
    # data_matrix = xyz.to_numpy()
    # idx = {col: i for i, col in enumerate(xyz.columns)}
    # kci_obj = CIT(data_matrix, "kci",
    #               KernelX='GaussianKernel',
    #               KernelY='GaussianKernel',
    #               KernelZ='GaussianKernel', approx=False, est_width='median')
    # zs = {'Z'}
    # pValue = kci_obj([idx['X']], [idx['Y']], [idx[z] for z in zs])
    # print(pValue)
    # print(data_matrix[:, 1].shape)
