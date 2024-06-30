from itertools import product

import networkx as nx
import numpy as np
import pandas as pd

from cddd.algos.PC_STABLE import pc_stable
from cddd.algos.PC_STABLE_with_CI_ORACLE import pc_stable_oracle
from cddd.cit import ci_test_factory
from cddd.evaluation import get_adj_mat, global_skeleton_metric_evaluation
from cddd.utils import safe_save_to_csv


def cached_pc_stable_experiment(Alphas, BN, Ks, ci_tester_name, reliability_criteria, sample_id, size_of_sampled_dataset,
                                working_dir):
    '''
    efficiently conduct performance experiment with pc-stable (skeleton discovery) using cache

    Args:
        Alphas: significance level for CIT to use
        BN: the name of Bayesian network
        Ks: threshold for deduce-dep
        ci_tester_name: the name of CIT
        reliability_criteria: reliability criterion used in structure learning
        sample_id: the ID of sampled dataset
        size_of_sampled_dataset: the size of sampled dataset
        working_dir: working directory

    Returns: None

    '''
    COLUMNS = ['sample_id', 'BN', 'size_of_sampled_dataset', 'reliability_criterion',
               'Accuracy', 'Precision', 'Recall', 'F1', 'CI_number']

    data, oracle_adj_mat = load_data(BN, sample_id, size_of_sampled_dataset, working_dir)
    ci_tester = ci_test_factory(ci_tester_name, data=data)

    # algorithm specific parameters
    for K, alpha, reliability_criterion in product(Ks, Alphas, reliability_criteria):
        estim_adj_mat, _, ci_number = pc_stable(data, alpha, reliability_criterion, K, ci_tester=ci_tester)
        accuracy, precision, recall, f1 = global_skeleton_metric_evaluation(oracle_adj_mat, estim_adj_mat)
        new_row = [sample_id, BN, size_of_sampled_dataset, reliability_criterion,
                   accuracy, precision, recall, f1, ci_number]

        result_file_path = f'{working_dir}/results/each_pc_stable_result_{alpha}_{K}.csv'
        safe_save_to_csv([new_row], COLUMNS, result_file_path)


def load_data(BN, sample_id, size_of_sampled_dataset, working_dir):
    '''
    load data for performance experiment

    Args:
        BN: the name of Bayesian network
        sample_id: the ID of sampled dataset
        size_of_sampled_dataset: the size of sampled dataset
        working_dir: working directory

    Returns:
        data : sampled dataset
        oracle_adj_mat = ground truth of DAG

    '''
    real_graph_path = f"{working_dir}/data/Ground_truth/{BN}_true.txt"
    data_path = f"{working_dir}/data/Sampled_datasets/{BN}_{size_of_sampled_dataset}_v{sample_id}.csv"
    data = pd.read_csv(data_path)
    _, all_number_para = np.shape(data)
    true_graph = nx.DiGraph(true_adj_mat := get_adj_mat(all_number_para, real_graph_path))
    # TODO replace to simple
    oracle_adj_mat = pc_stable_oracle(true_adj_mat, true_graph)

    _, kVar = np.shape(data)
    data.columns = [i for i in range(kVar)]

    return data, oracle_adj_mat
