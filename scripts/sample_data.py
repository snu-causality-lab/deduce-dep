import multiprocessing
import multiprocessing as mp
import os
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
from scipy.special import expit as sigmoid

from cddd.sampling import random_BN, random_CPTs, Erdos_Renyi_DAG, BayesianNetwork

CPTType = Dict[Tuple[int, ...], Sequence[float]]
T = TypeVar('T')
H = TypeVar('H', bound=Hashable)


def sample_synthetic_dataset(num_vars, edge_ratio, num_sampling, size, working_dir, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_edges = int(num_vars * edge_ratio)
    BN_name = f'synthetic_ER_{num_vars}_{num_edges}'
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


def sample_linear_sem(num_vars, edge_ratio, num_sampling, size, working_dir, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_edges = int(num_vars * edge_ratio)
    BN_name = f'Linear_{num_vars}_{num_edges}'

    # Generate graph
    B = nx.gnm_random_graph(num_vars, num_edges)
    B = nx.to_numpy_array(B)
    B = np.tril(B, -1)
    G = nx.DiGraph(B)

    # save ground truth file
    adj_mat_df = pd.DataFrame(B, columns=[i for i in range(num_vars)], dtype=np.int32)
    adj_mat_df_path = f'{working_dir}/data/Ground_truth/{BN_name}_true.txt'
    adj_mat_df.to_csv(adj_mat_df_path, sep=" ", header=None, index=False)

    # Generate weight matrix
    w_ranges = ((-0.7, -0.1), (0.1, 0.7))
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range

    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U

    # Sampling from linear SEM
    ordered_nodes = list(nx.topological_sort(G))
    scale_vec = np.ones(num_vars)

    for i in range(1, num_sampling + 1):
        X = np.zeros([size, num_vars])
        for j in ordered_nodes:
            parents = tuple(G.predecessors(j))
            X[:, j] = X[:, parents] @ W[parents, j] + np.random.normal(scale=scale_vec[j], size=size)

        sampled_data = pd.DataFrame(X, columns=[i for i in range(num_vars)], dtype=np.float32)
        sampled_data_path = f'{working_dir}/data/Sampled_datasets/{BN_name}_{size}_v{i}.csv'
        sampled_data.to_csv(sampled_data_path, index=False)


def sample_linear_sem_sf(num_vars, edge_to_attach, num_sampling, size, working_dir, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Generate graph (sf)
    B = nx.barabasi_albert_graph(num_vars, edge_to_attach)
    B = nx.to_numpy_array(B)
    B = np.tril(B, -1)
    G = nx.DiGraph(B)

    num_edges = len(G.edges)
    BN_name = f'Linear_{num_vars}_{num_edges}_sf'

    # save ground truth file
    adj_mat_df = pd.DataFrame(B, columns=[i for i in range(num_vars)], dtype=np.int32)
    adj_mat_df_path = f'{working_dir}/data/Ground_truth/{BN_name}_true.txt'
    adj_mat_df.to_csv(adj_mat_df_path, sep=" ", header=None, index=False)

    # Generate weight matrix
    w_ranges = ((-0.7, -0.1), (0.1, 0.7))
    W = np.zeros(B.shape)
    S = np.random.randint(len(w_ranges), size=B.shape)  # which range

    for i, (low, high) in enumerate(w_ranges):
        U = np.random.uniform(low=low, high=high, size=B.shape)
        W += B * (S == i) * U

    # Sampling from linear SEM
    ordered_nodes = list(nx.topological_sort(G))
    scale_vec = np.ones(num_vars)

    for i in range(1, num_sampling + 1):
        X = np.zeros([size, num_vars])
        for j in ordered_nodes:
            parents = tuple(G.predecessors(j))
            X[:, j] = X[:, parents] @ W[parents, j] + np.random.normal(scale=scale_vec[j], size=size)

        sampled_data = pd.DataFrame(X, columns=[i for i in range(num_vars)], dtype=np.float32)
        sampled_data_path = f'{working_dir}/data/Sampled_datasets/{BN_name}_{size}_v{i}.csv'
        sampled_data.to_csv(sampled_data_path, index=False)


def sample_nonlinear_sem(num_vars, edge_ratio, num_sampling, size, working_dir, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    num_edges = int(num_vars * edge_ratio)
    BN_name = f'Nonlinear_{num_vars}_{num_edges}'

    # Generate graph
    B = nx.gnm_random_graph(num_vars, num_edges)
    B = nx.to_numpy_array(B)
    B = np.tril(B, -1)
    G = nx.DiGraph(B)

    # save ground truth file
    adj_mat_df = pd.DataFrame(B, columns=[i for i in range(num_vars)], dtype=np.int32)
    adj_mat_df_path = f'{working_dir}/data/Ground_truth/{BN_name}_true.txt'
    adj_mat_df.to_csv(adj_mat_df_path, sep=" ", header=None, index=False)

    # Sampling from nonlinear SEM (using MLP)
    ordered_nodes = list(nx.topological_sort(G))
    scale_vec = np.ones(num_vars)

    for i in range(1, num_sampling + 1):
        X = np.zeros([size, num_vars])
        for j in ordered_nodes:
            parents = tuple(G.predecessors(j))

            hidden = 100
            pa_size = X[:, parents].shape[1]

            z = np.random.normal(scale=scale_vec[j], size=size)
            W1 = np.random.uniform(low=0.1, high=1.0, size=[pa_size, hidden])
            W1[np.random.rand(*W1.shape) < 0.5] *= -1
            W2 = np.random.uniform(low=0.1, high=1.0, size=hidden)
            W2[np.random.rand(hidden) < 0.5] *= -1
            X[:, j] = sigmoid(X[:, parents] @ W1) @ W2 + z

        sampled_data = pd.DataFrame(X, columns=[i for i in range(num_vars)], dtype=np.float32)
        sampled_data_path = f'{working_dir}/data/Sampled_datasets/{BN_name}_{size}_v{i}.csv'
        sampled_data.to_csv(sampled_data_path, index=False)


if __name__ == '__main__':
    # It will take about less than 2 minutes for Macbook M1 pro
    # To freshly generate data, delete Ground_truth and Sampled_datasets before running this script
    warnings.filterwarnings('ignore')

    # You will see (a lot of ignorable) warnings like
    # [d3blocks] >WARNING> Probability values don't exactly sum to 1. Differ by: -2.220446049250313e-16. Adjusting values.
    WORKING_DIR = os.path.expanduser('~/CD_DD')
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
    # generate data for Erdos-Renyi (previous version)
    Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(sample_synthetic_dataset)(
            num_vars, edge_ratio, num_sampling, size, WORKING_DIR, seed=i)
        for i, (num_vars, edge_ratio, size)
        in enumerate(product(nums_vars, edge_ratios, dataset_sizes))
    )

    edge_ratios = (1.8, 3.2)
    edges_to_attach = (2, 4)  # for scale-free networks
    dataset_sizes = (30, 50, 100, 150)

    print('generating linear SEM datasets.')  # post-submission : Linear SEM ER
    # generate data from linear SEM
    Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(sample_linear_sem)(
            num_vars, edge_ratio, num_sampling, size, WORKING_DIR, seed=i)
        for i, (num_vars, edge_ratio, size)
        in enumerate(product(nums_vars, edge_ratios, dataset_sizes))
    )

    print('generating nonlinear SEM datasets.')  # post-submission : Linear SEM
    # generate data from nonlinear SEM (using MLP)
    Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(sample_nonlinear_sem)(
            num_vars, edge_ratio, num_sampling, size, WORKING_DIR, seed=i)
        for i, (num_vars, edge_ratio, size)
        in enumerate(product(nums_vars, edge_ratios, dataset_sizes))
    )

    print('generating linear SEM SF datasets.')  # post-submission : Linear SEM SF
    # generate data from linear SEM SF version
    Parallel(n_jobs=multiprocessing.cpu_count())(
        delayed(sample_linear_sem_sf)(
            num_vars, edge_to_attach, num_sampling, size, WORKING_DIR, seed=i)
        for i, (num_vars, edge_to_attach, size)
        in enumerate(product(nums_vars, edges_to_attach, dataset_sizes))
    )
