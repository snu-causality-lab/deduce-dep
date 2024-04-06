import random
from collections import Counter

import networkx as nx
import numpy as np
import pandas as pd
from numpy.random import randint, choice

from cddd.cit import ci_test_factory
from cddd.deductive_reasoning import deduce_dep
from cddd.evaluation import get_adj_mat
from cddd.sampling import shuffled
from cddd.utils import safe_save_to_csv


def new_fn_experiment(BN, alpha, K, ci_tester_name, working_dir, dataset_size, sample_id):
    # experiments settings
    random.seed(0)
    ci_tester = ci_test_factory(ci_tester_name)

    # experiment results
    columns = ['BN', 'data_set_size', 'is_deductive_reasoning',
               'Accuracy', 'Precision', 'Recall', 'F1']
    result = []

    real_graph_path = f"{working_dir}/data/Ground_truth/{BN}_true.txt"
    data_path = f"{working_dir}/data/Sampled_datasets/{BN}_{dataset_size}_v"
    data = pd.read_csv(data_path + '1.csv')
    number_of_data, num_vars = np.shape(data)

    true_adj_mat = get_adj_mat(num_vars, real_graph_path)
    true_graph = nx.DiGraph(true_adj_mat)

    completePath = data_path + str(sample_id) + ".csv"
    data = pd.read_csv(completePath)
    number_of_data, num_vars = np.shape(data)
    data.columns = [i for i in range(num_vars)]

    result_mth = __fn_experiment_core(BN, K, alpha, data, num_vars, dataset_size, true_graph, 20, ci_tester=ci_tester)

    result.extend(result_mth)

    # write and save the experiment result as a csv file
    result_file_path = f'{working_dir}/results/new_fn_result_{alpha}_{K}.csv'
    safe_save_to_csv(result, columns, result_file_path)
    # print(f'new_fn_experiment {(BN, alpha, K, ci_tester_name, working_dir, dataset_size, sample_id)}')


def __fn_experiment_core(BN, K, alpha, data, num_vars, size_of_sampled_dataset, true_graph, n_repeats=20, ci_tester=None):
    result_mth = []
    sepsets = dict()
    consets = dict()
    add_ci_set = []
    results_stat = []
    results_deduce = []
    results_all = []
    # randomly check conditional independence from sample and d-separation, 0 <= ... <=N-2
    for _ in range(n_repeats):
        Vs = set([i for i in range(num_vars)])
        Zs = set(choice(num_vars, randint(2, min(5, num_vars - 1)), replace=False))
        X, Y, *_ = shuffled(Vs - Zs)

        truth = nx.d_separated(true_graph, {X}, {Y}, Zs)
        pval, _ = ci_tester.ci_test(data, X, Y, list(Zs))
        stat_estim = (pval > alpha)

        if pval > alpha:
            deduce_estim = not (deduce_dep(data, X, Y, list(Zs), K, alpha, add_ci_set, sepsets, consets, ci_tester=ci_tester))
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

        new_row = [BN, size_of_sampled_dataset, is_deductive_reasoning,
                   accuracy, precision, recall, f1]
        result_mth.append(new_row)
    return result_mth
