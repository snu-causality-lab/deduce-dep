import numpy as np

from cddd.evaluation import get_SHD
from cddd.utils import DAG_to_CPDAG

if __name__ == '__main__':
    # 0->1->2
    true_dag = np.zeros((3, 3), dtype=int)
    true_dag[0, 1] = 1
    true_dag[1, 2] = 1
    estim_dag = np.zeros((3, 3), dtype=int)
    estim_dag[0, 1] = estim_dag[1, 0] = 1
    estim_dag[1, 2] = estim_dag[2, 1] = 1

    true_cpdag = DAG_to_CPDAG(true_dag)
    # print(true_cpdag)
    # print(estim_dag)
    assert get_SHD(true_cpdag, estim_dag) == 0

    # a->b<-c
    true_dag = np.zeros((3, 3), dtype=int)
    true_dag[0, 1] = 1
    true_dag[2, 1] = 1
    # a-b-c
    estim_dag = np.zeros((3, 3), dtype=int)
    estim_dag[0, 1] = estim_dag[1, 0] = 1
    estim_dag[1, 2] = estim_dag[2, 1] = 1

    true_cpdag = DAG_to_CPDAG(true_dag)
    # print(true_cpdag)
    # print(estim_dag)
    assert get_SHD(true_cpdag, estim_dag) == 2

    # a->b<-c
    true_dag = np.zeros((3, 3), dtype=int)
    true_dag[0, 1] = 1
    true_dag[2, 1] = 1
    # a->b-c
    estim_dag = np.zeros((3, 3), dtype=int)
    estim_dag[0, 1] = 1
    estim_dag[1, 2] = estim_dag[2, 1] = 1

    true_cpdag = DAG_to_CPDAG(true_dag)
    # print(true_cpdag)
    # print(estim_dag)
    assert get_SHD(true_cpdag, estim_dag) == 1

    # a->b<-c
    true_dag = np.zeros((3, 3), dtype=int)
    true_dag[0, 1] = 1
    true_dag[2, 1] = 1
    # a->b->c
    estim_dag = np.zeros((3, 3), dtype=int)
    estim_dag[0, 1] = 1
    estim_dag[1, 2] = 1

    true_cpdag = DAG_to_CPDAG(true_dag)
    assert get_SHD(true_cpdag, estim_dag) == 1

    # a->b<-c
    true_dag = np.zeros((3, 3), dtype=int)
    true_dag[0, 1] = 1
    true_dag[2, 1] = 1
    # a<-b c
    estim_dag = np.zeros((3, 3), dtype=int)
    estim_dag[1, 0] = 1

    true_cpdag = DAG_to_CPDAG(true_dag)
    assert get_SHD(true_cpdag, estim_dag) == 2
