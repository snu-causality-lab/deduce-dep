import random
from collections import Counter

from numpy.random import randint, choice

from cddd.cit import ci_test_factory
from cddd.deductive_reasoning import deduce_dep
from cddd.sampling import random_BN, shuffled
from cddd.utils import safe_save_to_csv


def fn_experiment(working_dir, nums_vars=(10, 20, 30), times_vars=(1.2, 1.5, 2), dataset_sizes=(200, 500, 1000)):
    for alpha in (0.01, 0.05):
        __fn_experiment_core(working_dir, alpha, nums_vars, times_vars, dataset_sizes)


def __fn_experiment_core(working_dir, alpha, nums_vars, times_vars, dataset_sizes):
    assert alpha in {0.01, 0.05}
    alpha_str = {0.01: '001', 0.05: '005'}
    # experiments settings
    ci_tester = ci_test_factory('G2')
    # experiment results
    columns = ['num_vars', 'num_edges', 'data_set_size', 'is_deductive_reasoning',
               'Accuracy', 'Precision', 'Recall', 'F1']
    result = []

    for num_vars in nums_vars:
        for times_var in times_vars:
            for data_set_size in dataset_sizes:
                num_edges = int(times_var * num_vars)
                random.seed(0)

                for _ in range(50):
                    # randomly generate a Bayesian network
                    bn = random_BN(num_vars, num_edges)

                    # random sample
                    data = bn.sample(data_set_size)

                    sepsets = dict()
                    consets = dict()
                    add_ci_set = []

                    results_stat = []  # type:
                    results_deduce = []
                    results_all = []

                    # randomly check conditional independence from sample and d-separation, 0 <= ... <=N-2
                    for _ in range(20):
                        Zs = set(choice(num_vars, randint(2, min(5, num_vars - 1)), replace=False))
                        X, Y, *_ = shuffled(bn.Vs - Zs)

                        truth = bn.is_d_separated({X}, {Y}, Zs)
                        # pval, _ = cond_indep_test(data, X, Y, list(Zs))
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

                        new_row = [num_vars, num_edges, data_set_size, is_deductive_reasoning,
                                   accuracy, precision, recall, f1]
                        result.append(new_row)

    # write and save the experiment result as a csv file
    result_file_path = f'{working_dir}/results/fn_result_alpha_{alpha_str[alpha]}.csv'
    safe_save_to_csv(result, columns, result_file_path)
