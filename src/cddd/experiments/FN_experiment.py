import os
import random
from collections import Counter

import pandas as pd
from numpy.random import randint, choice

from cddd.cit import cond_indep_test
from cddd.deductive_reasoning import deduce_dep
from cddd.sampling import random_BN, shuffled


def fn_experiment(working_dir, nums_vars=(10, 20, 30), times_vars=(1.2, 1.5, 2), dataset_sizes=(200, 500, 1000)):
    for alpha in (0.01, 0.05):
        _fn_experiment_core(working_dir, alpha, nums_vars, times_vars, dataset_sizes)


def _fn_experiment_core(working_dir, alpha, nums_vars, times_vars, dataset_sizes):
    assert alpha in {0.01, 0.05}
    alpha_str = {0.01: '001', 0.05: '005'}
    # experiments settings

    # experiment results
    columns = ['num_vars', 'num_edges', 'data_set_size', 'is_deductive_reasoning',
               'Accuracy', 'Precision', 'Recall', 'F1']
    # columns = ['truth', 'stat_estim', 'deduce_estim']
    result = []

    for num_vars in nums_vars:
        for times_var in times_vars:
            for data_set_size in dataset_sizes:
                num_edges = int(times_var * num_vars)
                random.seed(0)
                # is_discrete = True

                for _ in range(50):
                    # randomly generate a Bayesian network
                    bn = random_BN(num_vars, num_edges)

                    # random sample
                    data = bn.sample(data_set_size)

                    sepsets = dict()
                    consets = dict()
                    add_ci_set = []

                    results_stat = []
                    results_deduce = []
                    results_all = []

                    # randomly check conditional independence from sample and d-separation, 0 <= ... <=N-2
                    for _ in range(20):
                        Zs = set(choice(num_vars, randint(2, min(5, num_vars - 1)), replace=False))
                        X, Y, *_ = shuffled(bn.Vs - Zs)

                        truth = bn.is_d_separated({X}, {Y}, Zs)
                        pval, _ = cond_indep_test(data, X, Y, list(Zs))
                        stat_estim = (pval > alpha)

                        if pval > alpha:
                            deduce_estim = not (deduce_dep(data, X, Y, list(Zs), 1, alpha, add_ci_set, sepsets, consets))
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

                        # print("<results_deduce>" if is_deductive_reasoning else "<results_stat>")
                        # print("TP, TN, FP, FN:", TP, TN, FP, FN)
                        # print("accuracy:", accuracy)
                        # print("precision:", precision)
                        # print("recall:", recall)
                        # print("f1:", f1)

                        new_row = [num_vars, num_edges, data_set_size, is_deductive_reasoning,
                                   accuracy, precision, recall, f1]
                        result.append(new_row)

    # write and save the experiment result as a csv file
    df_for_result = pd.DataFrame(result, columns=columns)
    result_file_path = f'{working_dir}/results/fn_result_alpha_{alpha_str[alpha]}.csv'

    if not os.path.exists(result_file_path):
        df_for_result.to_csv(result_file_path, mode='w')
    else:
        df_for_result.to_csv(result_file_path, mode='a', header=False)

# def naive_deduce_dep(data, X, Y, Zs, alpha):
#     for z in Zs:
#         remaining_Zs = list(set(Zs) - {z})
#         pval_XYZs, _ = cond_indep_test(data, X, Y, list(remaining_Zs))
#         pval_XzZs, _ = cond_indep_test(data, X, z, list(remaining_Zs))
#         pval_YzZs, _ = cond_indep_test(data, Y, z, list(remaining_Zs))
#
#         if (pval_XYZs < alpha) ^ ((pval_XzZs < alpha) and (pval_YzZs < alpha)):
#             return True
#     return False
