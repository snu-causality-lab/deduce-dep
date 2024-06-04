#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""A conditional independency test function for discrete data.

The code included in this package is logically copied and pasted from
the pcalg package for R developed by Markus Kalisch, Alain Hauser,
Martin Maechler, Diego Colombo, Doris Entner, Patrik Hoyer, Antti
Hyttinen, and Jonas Peters.

License: GPLv2
"""

import logging
import warnings
from typing import Tuple

import numpy as np
import statsmodels.api as sm2
from causallearn.utils.cit import CIT
from scipy.stats import chi2

warnings.filterwarnings('ignore')

_logger = logging.getLogger(__name__)


def g_square_dis(dm, x, y, s, levels):
    """G square test for discrete data.

    Args:
        dm: the data matrix to be used (as a numpy.ndarray).
        x: the first node (as an integer).
        y: the second node (as an integer).
        s: the set of neibouring nodes of x and y (as a set()).
        levels: levels of each column in the data matrix
            (as a list()).

    Returns:
        p_val: the p-value of conditional independence.
    """
    tlog = None

    def _calculate_tlog(x, y, s, levels, dm):
        prod_levels = np.prod(list(map(lambda x: levels[x], s)))
        nijk = np.zeros((levels[x], levels[y], prod_levels))
        s_size = len(s)
        z = []
        for z_index in range(s_size):
            z.append(s.pop())
            pass
        for row_index in range(dm.shape[0]):
            i = dm[row_index, x]
            j = dm[row_index, y]
            # k = []
            k_index = 0
            for s_index in range(s_size):
                if s_index == 0:
                    k_index += dm[row_index, z[s_index]]
                else:
                    lprod = np.prod(list(map(lambda x: levels[x], z[:s_index])))
                    k_index += (dm[row_index, z[s_index]] * lprod)
                    pass
                pass
            nijk[i, j, k_index] += 1
            pass
        nik = np.ndarray((levels[x], prod_levels))
        njk = np.ndarray((levels[y], prod_levels))
        for k_index in range(prod_levels):
            nik[:, k_index] = nijk[:, :, k_index].sum(axis=1)
            njk[:, k_index] = nijk[:, :, k_index].sum(axis=0)
            pass
        nk = njk.sum(axis=0)
        tlog = np.zeros((levels[x], levels[y], prod_levels))
        tlog.fill(np.nan)
        for k in range(prod_levels):
            tx = np.array([nik[:, k]]).T
            ty = np.array([njk[:, k]])
            tdijk = tx.dot(ty)
            tlog[:, :, k] = nijk[:, :, k] * nk[k] / tdijk
            pass
        return nijk, tlog

    _logger.debug('Edge %d -- %d with subset: %s' % (x, y, s))
    row_size = dm.shape[0]
    s_size = len(s)
    dof = ((levels[x] - 1) * (levels[y] - 1)
           * np.prod(list(map(lambda x: levels[x], s))))

    nijk = None
    if s_size < 5:
        if s_size == 0:
            nijk = np.zeros((levels[x], levels[y]))
            for row_index in range(row_size):
                i = dm[row_index, x]
                j = dm[row_index, y]
                nijk[i, j] += 1
                pass
            tx = np.array([nijk.sum(axis=1)]).T
            ty = np.array([nijk.sum(axis=0)])
            tdij = tx.dot(ty)
            tlog = nijk * row_size / tdij
            pass
        if s_size > 0:
            nijk, tlog = _calculate_tlog(x, y, s, levels, dm)
            pass
        pass
    else:
        # s_size >= 5
        nijk = np.zeros((levels[x], levels[y], 1))
        i = dm[0, x]
        j = dm[0, y]
        k = []
        for z in s:
            k.append(dm[:, z])
            pass
        k = np.array(k).T
        parents_count = 1
        parents_val = np.array([k[0, :]])
        nijk[i, j, parents_count - 1] = 1
        for it_sample in range(1, row_size):
            is_new = True
            i = dm[it_sample, x]
            j = dm[it_sample, y]
            tcomp = parents_val[:parents_count, :] == k[it_sample, :]
            for it_parents in range(parents_count):
                if np.all(tcomp[it_parents, :]):
                    nijk[i, j, it_parents] += 1
                    is_new = False
                    break
                pass
            if is_new is True:
                parents_count += 1
                parents_val = np.r_[parents_val, [k[it_sample, :]]]
                nnijk = np.zeros((levels[x], levels[y], parents_count))
                for p in range(parents_count - 1):
                    nnijk[:, :, p] = nijk[:, :, p]
                    pass
                nnijk[i, j, parents_count - 1] = 1
                nijk = nnijk
                pass
            pass
        nik = np.ndarray((levels[x], parents_count))
        njk = np.ndarray((levels[y], parents_count))
        for k_index in range(parents_count):
            nik[:, k_index] = nijk[:, :, k_index].sum(axis=1)
            njk[:, k_index] = nijk[:, :, k_index].sum(axis=0)
            pass
        nk = njk.sum(axis=0)
        tlog = np.zeros((levels[x], levels[y], parents_count))
        tlog.fill(np.nan)
        for k in range(parents_count):
            tx = np.array([nik[:, k]]).T
            ty = np.array([njk[:, k]])
            tdijk = tx.dot(ty)
            tlog[:, :, k] = nijk[:, :, k] * nk[k] / tdijk
            pass
        pass
    log_tlog = np.log(tlog)
    G2 = np.nansum(2 * nijk * log_tlog)
    _logger.debug('G2 = %f' % G2)
    if dof == 0:
        # dof can be 0 when levels[x] or levels[y] is 1, which is
        # the case that the values of columns x or y are all 0.
        p_val = 1
        G2 = 0
    else:
        p_val = chi2.sf(G2, dof)
    _logger.info('p_val = %s' % str(p_val))

    dep = abs(G2)
    return p_val, dep


def g2_test_dis(data_matrix, x, y, s, **kwargs):
    s1 = sorted([i for i in s])
    data_matrix = np.array(data_matrix, dtype=int)

    if 'levels' in kwargs:
        levels = kwargs['levels']
    else:
        levels = np.amax(data_matrix, axis=0) + 1

    return g_square_dis(data_matrix, x, y, s1, levels)


class CITester:
    def __init__(self, **kwargs):
        ...

    def ci_test(self, data, X, Y, cond_set=frozenset()) -> Tuple[float, float]:
        p_val, dependency = 0, 0
        return p_val, dependency


class G2Tester(CITester):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = None, None, None
        if 'data' in kwargs:
            self.data = kwargs['data']
        self.cache = dict()

    def ci_test(self, data, X, Y, cond_set=frozenset()):
        if data is not None and data is not self.data:
            pval, dep = g2_test_dis(data, X, Y, cond_set)
            return pval, dep
        return self.cached_ci_test(tuple(sorted([X, Y])), frozenset(cond_set))

    def cached_ci_test(self, XY, cond_set=frozenset()):
        if (XY, cond_set) in self.cache:
            return self.cache[(XY, cond_set)]
        X, Y = XY
        pval, dep = g2_test_dis(self.data, X, Y, cond_set)

        self.cache[(XY, cond_set)] = (pval, dep)
        return self.cache[(XY, cond_set)]


class PartialCorrelation(CITester):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data = None, None, None
        if 'data' in kwargs:
            self.data = kwargs['data']
        self.cache = dict()

    def ci_test(self, data, X, Y, cond_set=frozenset()):
        if data is not None and data is not self.data:
            result = sm2.OLS(data[X], sm2.add_constant(data[[Y] + list(cond_set)])).fit()
            p_values = result.summary2().tables[1]['P>|t|']
            pval = p_values[Y]
            # Note that X and Y are symmetric
            return pval, -pval
        return self.cached_ci_test(tuple(sorted([X, Y])), frozenset(cond_set))

    def cached_ci_test(self, XY, cond_set=frozenset()):
        if (XY, cond_set) in self.cache:
            return self.cache[(XY, cond_set)]
        X, Y = XY

        result = sm2.OLS(self.data[X], sm2.add_constant(self.data[[Y] + list(cond_set)])).fit()
        p_values = result.summary2().tables[1]['P>|t|']
        pval = p_values[Y]

        self.cache[(XY, cond_set)] = (pval, -pval)
        return self.cache[(XY, cond_set)]


class KernelCITest(CITester):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.data, self.kci_obj, self.idx = None, None, None
        if 'data' in kwargs:
            self.data = kwargs['data']
            data_matrix = self.data.to_numpy()
            self.idx = {col: i for i, col in enumerate(self.data.columns)}
            self.kci_obj = CIT(data_matrix, "kci",
                               KernelX='GaussianKernel',
                               KernelY='GaussianKernel',
                               KernelZ='GaussianKernel', approx=False, est_width='median')
        self.cache = dict()

    def ci_test(self, data, X, Y, cond_set=frozenset()):
        if data is not None and data is not self.data:
            data_matrix = data.to_numpy()
            idx = {col: i for i, col in enumerate(data.columns)}
            kci_obj = CIT(data_matrix, "kci",
                          KernelX='GaussianKernel',
                          KernelY='GaussianKernel',
                          KernelZ='GaussianKernel', approx=False, est_width='median')
            pval = kci_obj([idx[X]], [idx[Y]], [idx[z] for z in cond_set])
            return pval, -pval

        return self.cached_ci_test(tuple(sorted([X, Y])), frozenset(cond_set))

    def cached_ci_test(self, XY, cond_set=frozenset()):
        if (XY, cond_set) in self.cache:
            return self.cache[(XY, cond_set)]
        X, Y = XY
        pval = self.kci_obj([self.idx[X]], [self.idx[Y]], [self.idx[z] for z in cond_set])

        self.cache[(XY, cond_set)] = (pval, -pval)
        return self.cache[(XY, cond_set)]


def ci_test_factory(name, **kwargs):
    if name == 'G2':
        return G2Tester(**kwargs)
    elif name == 'ParCorr':
        return PartialCorrelation(**kwargs)
    elif name == 'KCI':
        return KernelCITest(**kwargs)
    else:
        raise AssertionError(f'unknown CI tester: {name}')
