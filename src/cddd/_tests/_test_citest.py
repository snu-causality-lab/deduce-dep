import networkx as nx
import numpy as np
import pandas as pd

from cddd.cit import PartialCorrelation, G2Tester, KernelCITest
from cddd.sampling import random_CPTs, BayesianNetwork


def check_KCI():
    repeats = 20
    # n = 100
    # beta = 0.2
    # z -> x -> w -> y <- z
    # z = 0.5 * np.random.randn(n)
    for n in [50, 100]:
        avg = 0

        for _ in range(repeats):
            x = np.random.randn(n)
            z = x + np.random.randn(n)
            y = z + np.random.randn(n)

            xyz = pd.DataFrame(np.vstack([x, y, z]).T)
            xyz.columns = ('X', 'Y', 'Z')
            citt = KernelCITest(data=xyz)

            pval, dep = citt.ci_test(xyz, 'X', 'Y', {'Z'})
            pval, dep = citt.ci_test(xyz, 'X', 'Y', {'Z'})
            pval, dep = citt.ci_test(xyz, 'Y', 'X', {'Z'})
            avg += pval < 0.05
        print(f'H0: {n=}, {avg / repeats=}')
    #
    # beta = 0.2
    # z -> x -> w -> y <- z
    # z = 0.5 * np.random.randn(n)
    for n in [100, 200]:
        for beta in [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]:
            avg = 0
            repeats = 500
            for _ in range(repeats):
                z = np.random.randn(n)
                x = z + np.random.randn(n)
                y = z + beta * x + np.random.randn(n)

                citt = PartialCorrelation()

                # xyz = pd.DataFrame(np.vstack([x, y, w, z]).T)
                xyz = pd.DataFrame(np.vstack([x, y, z]).T)
                # xyz.columns = ('X', 'Y', 'W', 'Z')
                xyz.columns = ('X', 'Y', 'Z')
                # print(xyz.columns)
                pval, dep = citt.ci_test(xyz, 'X', 'Y', {'Z'})
                avg += pval < 0.05
            print(f'{n=}, {beta=}, {avg / repeats=}')


def check_parcorr():
    repeats = 10
    # n = 100
    # beta = 0.2
    # z -> x -> w -> y <- z
    # z = 0.5 * np.random.randn(n)
    for n in [10, 20, 30, 50, 100, 200, 500, 1000]:
        avg = 0
        for _ in range(repeats):
            x = np.random.randn(n)
            z = x + np.random.randn(n)
            y = z + np.random.randn(n)

            xyz = pd.DataFrame(np.vstack([x, y, z]).T)
            parcorr = PartialCorrelation(data=xyz)
            xyz.columns = ('X', 'Y', 'Z')
            pval, dep = parcorr.ci_test(xyz, 'X', 'Y', {'Z'})
            pval, dep = parcorr.ci_test(xyz, 'X', 'Y', {'Z'})
            pval, dep = parcorr.ci_test(xyz, 'Y', 'X', {'Z'})
            avg += pval < 0.05
        print(f'H0: {n=}, {avg / repeats=}')
    #
    # beta = 0.2
    # z -> x -> w -> y <- z
    # z = 0.5 * np.random.randn(n)
    for n in [100, 200, 500]:
        for beta in [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3]:
            avg = 0
            for _ in range(repeats):
                z = np.random.randn(n)
                x = z + np.random.randn(n)
                y = z + beta * x + np.random.randn(n)

                parcorr = PartialCorrelation()

                # xyz = pd.DataFrame(np.vstack([x, y, w, z]).T)
                xyz = pd.DataFrame(np.vstack([x, y, z]).T)
                # xyz.columns = ('X', 'Y', 'W', 'Z')
                xyz.columns = ('X', 'Y', 'Z')
                # print(xyz.columns)
                pval, dep = parcorr.ci_test(xyz, 'X', 'Y', {'Z'})
                avg += pval < 0.05
            print(f'{n=}, {beta=}, {avg / repeats=}')


def check_g2():
    # n = 100
    # beta = 0.2
    # z -> x -> w -> y <- z
    # z = 0.5 * np.random.randn(n)
    for n in [10, 20, 30, 50, 100, 200]:
        avg = 0
        repeats = 500
        for _ in range(repeats):
            graph = nx.DiGraph()
            # X 0,Y 1,Z 2
            graph.add_edge(2, 0)
            graph.add_edge(2, 1)
            # graph.add_edge(0, 1)
            CPTs = random_CPTs(graph)
            bn = BayesianNetwork(graph, CPTs)
            xyz = bn.sample(n)
            citester = G2Tester(data=xyz)
            pval, dep = citester.ci_test(xyz, 0, 1, {2})
            pval, dep = citester.ci_test(xyz, 0, 1, {2})
            pval, dep = citester.ci_test(xyz, 0, 1, {2})
            avg += pval < 0.05
        print(f'H0: {n=}, {avg / repeats=}')
    #
    # beta = 0.2
    # z -> x -> w -> y <- z
    # z = 0.5 * np.random.randn(n)
    for n in [100, 200, 500]:
        avg = 0
        repeats = 500
        for _ in range(repeats):
            graph = nx.DiGraph()
            # X 0,Y 1,Z 2
            graph.add_edge(2, 0)
            graph.add_edge(2, 1)
            graph.add_edge(0, 1)
            CPTs = random_CPTs(graph)
            bn = BayesianNetwork(graph, CPTs)
            xyz = bn.sample(n)
            pval, dep = citester.ci_test(xyz, 0, 1, {2})
            avg += pval < 0.05
        print(f'{n=}, {avg / repeats=}')


if __name__ == '__main__':
    # check_parcorr()
    check_KCI()
    # check_g2()
