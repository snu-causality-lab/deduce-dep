import random
import numpy as np
import pandas as pd
import networkx as nx
from collections import Counter

from numpy.random import randint, choice

from cddd.cit import ci_test_factory
from cddd.deductive_reasoning import deduce_dep
from cddd.evaluation import get_adj_mat
from cddd.sampling import random_BN, shuffled
from cddd.utils import safe_save_to_csv


def correction_experiment(working_dir, num_vars, time_vars, sampling_number, alpha, dataset_size):
    __correction_experiment_core(working_dir, num_vars, time_vars, sampling_number, alpha, dataset_size)


def __correction_experiment_core(working_dir, num_vars, time_vars, sampling_number, alpha, dataset_size):
    assert alpha in {0.01, 0.05}
    alpha_str = {0.01: '001', 0.05: '005'}
    # experiments settings
    ci_tester = ci_test_factory('G2')
    # experiment results
    columns = ['num_vars', 'num_edges', 'data_set_size', 'is_deductive_reasoning',
               'Accuracy', 'Precision', 'Recall', 'F1']
    result = []

    num_edges = int(time_vars * num_vars)
    random.seed(0)

    # get ground truth graph
    BN_name = f'synthetic_ER_{num_vars}_{num_edges}_{sampling_number}'
    real_graph_path = working_dir + f'/data/Ground_truth/{BN_name}_true.txt'
    true_adj_mat = get_adj_mat(num_vars, real_graph_path)
    true_graph = nx.DiGraph(true_adj_mat)

    # get sampled_data
    data = pd.read_csv(working_dir + f'/data/Sampled_datasets/{BN_name}_{dataset_size}_v1.csv')

    sepsets = dict()
    consets = dict()
    add_ci_set = []

    results_stat = []  # type:
    results_deduce = []
    results_all = []

    # randomly check conditional independence from sample and d-separation, 0 <= ... <=N-2
    for _ in range(20):
        np.random.seed(42)
        Zs = set(choice(num_vars, randint(2, min(5, num_vars - 1)), replace=False))
        X, Y, *_ = shuffled(set(true_graph.nodes) - Zs)

        truth = nx.d_separated(true_graph, {X}, {Y}, Zs)
        pval, _ = ci_tester.ci_test(data, X, Y, list(Zs))
        stat_estim = (pval > alpha)

        if pval > alpha:
            deduce_estim = not (deduce_dep(data, X, Y, list(Zs), 1, alpha, add_ci_set, sepsets, consets, ci_tester=ci_tester))
        else:
            deduce_estim = False

        results_stat.append((truth, stat_estim))
        results_deduce.append((truth, deduce_estim))
        results_all.append((truth, stat_estim, deduce_estim))

    for i, results in enumerate([results_stat, results_deduce]):
        counter = Counter(results)
        TP = counter.get((False, False), 0)
        TN = counter.get((True, True), 0)
        FP = counter.get((True, False), 0)
        FN = counter.get((False, True), 0)

        accuracy = (TP + TN) / (TP + FN + FP + TN)
        if (TP + FP) == 0:
            precision = 0
        else:
            precision = TP / (TP + FP)
        if (TP + FN) == 0:
            recall = 0
        else:
            recall = TP / (TP + FN)
        if (precision + recall) == 0:
            f1 = 0
        else:
            f1 = (2 * precision * recall) / (precision + recall)

        is_deductive_reasoning = (i != 0)

        new_row = [num_vars, num_edges, dataset_size, is_deductive_reasoning,
                   accuracy, precision, recall, f1]
        result.append(new_row)

    # write and save the experiment result as a csv file
    result_file_path = f'{working_dir}/results/corr_result_alpha_{alpha_str[alpha]}.csv'
    safe_save_to_csv(result, columns, result_file_path)
