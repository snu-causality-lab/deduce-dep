import numpy as np
import pandas as pd
import networkx as nx

from cddd.algos.algo_utils import estimate_cpdag

from cddd.evaluation import get_SHD

if __name__ == '__main__':
    # collider check
    # a->b<-c
    true_dag = np.zeros((3, 3), dtype=int)
    true_dag[0, 1] = 1
    true_dag[2, 1] = 1

    estim_dag = np.zeros((3,3), dtype =int)
    estim_dag[0, 1] = 1
    estim_dag[1, 0] = 1
    estim_dag[2, 1] = 1
    estim_dag[1, 2] = 1

    estim_G = pd.DataFrame(estim_dag, columns=[i for i in range(3)], dtype=int)
    estim_G = nx.DiGraph(estim_G)
    sepset = {(0, 2): ()}

    estim_dag = estimate_cpdag(estim_G, sepset)
    estim_dag = nx.to_numpy_array(estim_dag)

    assert np.all(true_dag == estim_dag) == True

    # Meek's rule 1 check
    # a->b-c
    true_dag = np.zeros((3, 3), dtype=int)
    true_dag[0, 1] = 1
    true_dag[1, 2] = 1

    estim_dag = np.zeros((3, 3), dtype=int)
    estim_dag[0, 1] = 1
    estim_dag[1, 2] = 1
    estim_dag[2, 1] = 1

    estim_G = pd.DataFrame(estim_dag, columns=[i for i in range(3)], dtype=int)
    estim_G = nx.DiGraph(estim_G)
    sepset = {(0, 2) : [1]}

    estim_dag = estimate_cpdag(estim_G, sepset)
    estim_dag = nx.to_numpy_array(estim_dag)

    assert np.all(true_dag == estim_dag) == True

    # Meek's rule 2 check
    # Orient a-b into a->b whenever there is a chain a -> c -> b
    true_dag = np.zeros((3, 3), dtype=int)
    true_dag[0, 1] = 1
    true_dag[0, 2] = 1
    true_dag[2, 1] = 1

    estim_dag = np.zeros((3, 3), dtype=int)
    estim_dag[0, 1] = 1
    estim_dag[1, 0] = 1
    estim_dag[0, 2] = 1
    estim_dag[2, 1] = 1

    estim_G = pd.DataFrame(estim_dag, columns=[i for i in range(3)], dtype=int)
    estim_G = nx.DiGraph(estim_G)
    sepset = {}

    estim_dag = estimate_cpdag(estim_G, sepset)
    estim_dag = nx.to_numpy_array(estim_dag)

    assert np.all(true_dag == estim_dag) == True

    # Meek's rule 3 check
    # Orient a-b into a->b whenever there are two chains a-c->b and a-d->b
    # s.t. c and d are nonadjacent.

    true_dag = np.zeros((4, 4), dtype=int)
    true_dag[0, 1] = 1
    true_dag[0, 2] = 1
    true_dag[2, 0] = 1
    true_dag[0, 3] = 1
    true_dag[3, 0] = 1
    true_dag[2, 1] = 1
    true_dag[3, 1] = 1

    estim_dag = np.zeros((4, 4), dtype=int)
    estim_dag[0, 1] = 1
    estim_dag[1, 0] = 1
    estim_dag[0, 2] = 1
    estim_dag[2, 0] = 1
    estim_dag[0, 3] = 1
    estim_dag[3, 0] = 1
    estim_dag[2, 1] = 1
    estim_dag[3, 1] = 1

    estim_G = pd.DataFrame(estim_dag, columns=[i for i in range(4)], dtype=int)
    estim_G = nx.DiGraph(estim_G)
    sepset = {(2, 3) : [0]}
    # sepset = {(2, 3): []} # all satisfied for two cases

    estim_dag = estimate_cpdag(estim_G, sepset)
    estim_dag = nx.to_numpy_array(estim_dag)

    assert np.all(true_dag == estim_dag) == True

    # Meek's rule 4 check
    # Orient a-b into a->b whenever there are two chains a-c->d and c->d->b
    # s.t. c and b are nonadjacent and a and d are adjacent.

    true_dag = np.zeros((4, 4), dtype=int)
    true_dag[0, 1] = 1

    true_dag[0, 2] = 1
    true_dag[2, 0] = 1

    # edge with star
    true_dag[0, 3] = 1
    true_dag[3, 0] = 1

    true_dag[2, 3] = 1
    true_dag[3, 1] = 1

    estim_dag = np.zeros((4, 4), dtype=int)
    estim_dag[0, 1] = 1
    estim_dag[1, 0] = 1

    estim_dag[0, 2] = 1
    estim_dag[2, 0] = 1

    # edge with star
    estim_dag[0, 3] = 1
    estim_dag[3, 0] = 1

    estim_dag[2, 3] = 1
    estim_dag[3, 1] = 1

    estim_G = pd.DataFrame(estim_dag, columns=[i for i in range(4)], dtype=int)
    estim_G = nx.DiGraph(estim_G)
    sepset = {(1, 2): [0, 3]}

    estim_dag = estimate_cpdag(estim_G, sepset)
    estim_dag = nx.to_numpy_array(estim_dag)

    assert np.all(true_dag == estim_dag) == True






