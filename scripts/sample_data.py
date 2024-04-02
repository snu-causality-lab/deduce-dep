import multiprocessing
import multiprocessing as mp
import random
import warnings
from itertools import product
from pathlib import Path
from typing import Dict, Tuple, Sequence, TypeVar, Hashable

import bnlearn as bn
import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from cddd.sampling import random_BN, random_CPTs, Erdos_Renyi_DAG, BayesianNetwork

CPTType = Dict[Tuple[int, ...], Sequence[float]]
T = TypeVar('T')
H = TypeVar('H', bound=Hashable)


def sample_synthetic_dataset(num_vars, num_edges, num_sampling, size, working_dir, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    BN_name = f'synthetic_{num_vars}_{num_edges}'
    for i in range(1, num_sampling + 1):
        # randomly generate a Bayesian network
        bn = random_BN(num_vars, num_edges)

        # save ground truth file
        adj_mat = nx.adjacency_matrix(bn.G)
        adj_mat = adj_mat.todense()
        adj_mat_df = pd.DataFrame(adj_mat, columns=[i for i in range(num_vars)])
        adj_mat_df_path = f'{working_dir}/data/Ground_truth/{BN_name}_true.txt'
        adj_mat_df.to_csv(adj_mat_df_path, sep=" ", header=None, index=False)

        # random sample
        sampled_data = bn.sample(size)
        # sampled_data_path = f'{working_dir}/data/Sampled_datasets/{BN_name}_' + str(size) + '_v' + str(i) + '.csv'
        sampled_data_path = f'{working_dir}/data/Sampled_datasets/{BN_name}_{size}_v{i}.csv'
        sampled_data.to_csv(sampled_data_path, index=False)


def sample_ER_dataset(num_vars, edge_ratio, num_sampling, size, working_dir, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_edges = int(num_vars * edge_ratio)
    BN_name = f'ER_{num_vars}_{num_edges}'
    # randomly generate a Bayesian network
    graph = Erdos_Renyi_DAG(num_vars, num_edges)
    CPTs = random_CPTs(graph)
    bn = BayesianNetwork(graph, CPTs)

    # save ground truth file
    adj_mat = nx.to_numpy_array(bn.G)
    adj_mat_df = pd.DataFrame(adj_mat, columns=[i for i in range(num_vars)], dtype=np.int32)
    adj_mat_df_path = f'{working_dir}/data/Ground_truth/{BN_name}_true.txt'
    adj_mat_df.to_csv(adj_mat_df_path, sep=" ", header=None, index=False)

    for i in range(1, num_sampling + 1):
        # random sample
        sampled_data = bn.sample(size)
        # sampled_data_path = f'{working_dir}/data/Sampled_datasets/{BN_name}_' + str(size) + '_v' + str(i) + '.csv'
        sampled_data_path = f'{working_dir}/data/Sampled_datasets/{BN_name}_{size}_v{i}.csv'
        sampled_data.to_csv(sampled_data_path, index=False)


# Loading DAG with model parameters from bif file.

def sample_from_bif(working_dir, bif_bn, sizes, sampling_number):
    bif_file = f'{working_dir}/data/bifs/{bif_bn}.bif'
    model = bn.import_DAG(bif_file, CPD=True, verbose=0)

    adjmat = bn.dag2adjmat(model['model'])
    adjmat.columns = [i for i in range(adjmat.shape[0])]
    adjmat.index = [i for i in range(adjmat.shape[0])]
    adjmat = adjmat.replace({False: 0, True: 1})
    adjmat.to_csv(f'{working_dir}/data/Ground_truth/{bif_bn}_true.txt', sep=" ", header=False, index=False)

    # The information of the benchmark bayesian network
    # number_of_nodes = adjmat.shape[0]
    # The representation of adjacency matrix
    # labels = [i for i in range(number_of_nodes)]
    # adjmat_for_graph = pd.DataFrame(adjmat, index=labels, columns=labels)
    # The representation of DAG
    # G = nx.DiGraph(adjmat_for_graph)
    # nx.draw(G, with_labels=True)

    def data_sampling(size, sampling_number):
        for i in range(1, sampling_number + 1):
            random.seed(hash((bif_bn, size, i)) & (2 ** 32 - 1))
            np.random.seed(hash((bif_bn, size, i)) & (2 ** 32 - 1))

            sampled_dataframe = bn.sampling(model, n=size, methodtype='bayes', verbose=0)
            sample_data_path = f'{working_dir}/data/Sampled_datasets/{bif_bn}_{size}_v{i}.csv'
            sampled_dataframe.to_csv(sample_data_path, index=False)

    # Sampling data from the BN
    Parallel(n_jobs=mp.cpu_count())(delayed(data_sampling)(size, sampling_number) for size in sizes)


if __name__ == '__main__':
    # It will take about ~55 secs for Macbook M1 pro
    # To freshly generate data, delete Ground_truth and Sampled_datasets before running this script
    warnings.filterwarnings('ignore')

    # You will see (a lot of ignorable) warnings like
    # [d3blocks] >WARNING> Probability values don't exactly sum to 1. Differ by: -2.220446049250313e-16. Adjusting values.
    WORKING_DIR = '/Users/sanghacklee/Dropbox/python_projs/CD_DD'
    graph_dir = f'{WORKING_DIR}/data/Ground_truth'
    data_dir = f'{WORKING_DIR}/data/Sampled_datasets'

    for directory in (graph_dir, data_dir):
        Path(directory).mkdir(parents=True, exist_ok=True)

    nums_vars = (10, 20, 30)
    edge_ratios = (1.2, 1.5, 2)
    num_sampling = 30
    dataset_sizes = (200, 500, 1000, 2000)

    # generate data for BNLearn models
    print('generating datasets for known BNLearn graphs...')
    BIF_BNs = ('alarm', 'asia', 'child', 'insurance', 'sachs', 'water')
    for bif_bn in BIF_BNs:
        print(f'    {bif_bn} ... ')
        sample_from_bif(WORKING_DIR, bif_bn, dataset_sizes, num_sampling)

    print('generating new-style random datasets.')  # post-submission
    # generate data for Erdos-Renyi
    Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(sample_ER_dataset)(
            num_vars, edge_ratio, num_sampling, size, WORKING_DIR, seed=i)
        for i, (num_vars, edge_ratio, size)
        in enumerate(product(nums_vars, edge_ratios, dataset_sizes))
    )

    print('generating previous style random datasets.')  # in the submission
    sample_synthetic_dataset(5, 10, 10, 200, WORKING_DIR, seed=0)
