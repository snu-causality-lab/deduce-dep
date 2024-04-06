import os
from itertools import product

import networkx as nx
import numpy as np
import pandas as pd
from filelock import FileLock
from joblib import Parallel, delayed

from cddd.algos.PC_STABLE import pc_stable
from cddd.algos.PC_STABLE_with_CI_ORACLE import pc_stable_oracle
from cddd.cit import ci_test_factory
from cddd.evaluation import get_adj_mat, global_skeleton_metric_evaluation

COLUMNS = ['sample_id', 'BN', 'size_of_sampled_dataset', 'reliability_criterion',
           'Accuracy', 'Precision', 'Recall', 'F1', 'CI_number']


def cached_pc_stable_experiment(working_dir, BNs, CITs, Ks, Alphas, dataset_sizes, num_sampling, reliability_criteria, n_jobs):
    # data-specific parameters
    Parallel(n_jobs=n_jobs)(delayed(__run_per_data)(Alphas, BN, Ks, ci_tester_name, reliability_criteria, sample_id, size_of_sampled_dataset, working_dir) for (BN, ci_tester_name), size_of_sampled_dataset, sample_id in product(list(zip(BNs, CITs)),
                                                                                                                                                                                                                                   dataset_sizes,
                                                                                                                                                                                                                                   list(range(1,
                                                                                                                                                                                                                                              num_sampling + 1))))


def __run_per_data(Alphas, BN, Ks, ci_tester_name, reliability_criteria, sample_id, size_of_sampled_dataset, working_dir):
    data, oracle_adj_mat = load_data(BN, sample_id, size_of_sampled_dataset, working_dir)
    ci_tester = ci_test_factory(ci_tester_name, data=data)
    result = []
    # algorithm specific parameters
    for K, alpha, reliability_criterion in product(Ks, Alphas, reliability_criteria):
        estim_adj_mat, _, ci_number = pc_stable(data, alpha, reliability_criterion, K, ci_tester=ci_tester)
        accuracy, precision, recall, f1 = global_skeleton_metric_evaluation(oracle_adj_mat, estim_adj_mat)
        new_row = [sample_id, BN, size_of_sampled_dataset, reliability_criterion,
                   accuracy, precision, recall, f1, ci_number]
        result.append(new_row)
    __save_to_csv(COLUMNS, K, alpha, result, working_dir)


def load_data(BN, sample_id, size_of_sampled_dataset, working_dir):
    real_graph_path = f"{working_dir}/data/Ground_truth/{BN}_true.txt"
    data_path = f"{working_dir}/data/Sampled_datasets/{BN}_{size_of_sampled_dataset}_v{sample_id}.csv"
    data = pd.read_csv(data_path)
    _, all_number_Para = np.shape(data)
    true_graph = nx.DiGraph(true_adj_mat := get_adj_mat(all_number_Para, real_graph_path))
    # TODO replace to simple
    oracle_adj_mat = pc_stable_oracle(true_adj_mat, true_graph)

    _, kVar = np.shape(data)
    data.columns = [i for i in range(kVar)]

    return data, oracle_adj_mat


def __save_to_csv(COLUMNS, K, alpha, result, working_dir):
    df_for_result = pd.DataFrame(result, columns=COLUMNS)
    result_file_path = f'{working_dir}/results/each_pc_stable_result_{alpha}_{K}.csv'
    lock = FileLock(result_file_path)
    with lock:
        if not os.path.exists(result_file_path):
            df_for_result.to_csv(result_file_path, mode='w')
        else:
            df_for_result.to_csv(result_file_path, mode='a', header=False)
