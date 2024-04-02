import os
from itertools import product

import pandas as pd

from cddd.evaluation import pc_stable_evaluation


def pc_stable_experiment(BN, working_dir, dataset_sizes=(200, 500, 1000, 2000), sampling_number=30):
    reliability_criterions = ['no', 'deductive_reasoning']
    # constants
    isdiscrete = True
    Ks = [0, 1, 2]
    Alphas = [0.05, 0.01]
    COLUMNS = ['BN', 'size_of_sampled_dataset', 'reliability_criterion',
               'Accuracy', 'Precision', 'Recall', 'F1', 'CI_number', 'Time',
               'Precision_std', 'Recall_std', 'F1_std', 'CI_number_std', 'Time_std']

    for K, alpha in product(Ks, Alphas):
        # experiment results
        _pc_stable_experiment_core(BN, COLUMNS, K, alpha, dataset_sizes, isdiscrete, reliability_criterions, sampling_number, working_dir)


def _pc_stable_experiment_core(BN, COLUMNS, K, alpha, dataset_sizes, isdiscrete, reliability_criterions, sampling_number, working_dir):
    result = []
    for size_of_sampled_dataset in dataset_sizes:
        real_graph_path = f"{working_dir}/data/Ground_truth/" + BN + "_true.txt"
        data_path = f"{working_dir}/data/Sampled_datasets/" + BN + "_" + str(size_of_sampled_dataset) + '_v'

        file_number = sampling_number

        for reliability_criterion in reliability_criterions:
            Accuracy, Precision, Recall, F1, CI_number, Time, \
                Precision_std, Recall_std, F1_std, CI_number_std, Time_std = pc_stable_evaluation(data_path, real_graph_path, isdiscrete, file_number, alpha, reliability_criterion, K=K)

            # print("------------------------------------------------------")
            # print("the BN of dataset is:", BN)
            # print("the size of dataset is:", size_of_sampled_dataset)
            # print("Reliability criterion:", reliability_criterion)
            # print()
            # print("Precision is: " + str("%.2f" % Precision), "with std: " + str("%.2f" % Precision_std))
            # print("Recall is: " + str("%.2f" % Recall), "with std: " + str("%.2f" % Recall_std))
            # print("F1 is: " + str("%.2f" % F1), "with std: " + str("%.2f" % F1_std))
            # print("CI_number is: " + str("%.2f" % CI_number) + " with std: " + str("%.2f" % CI_number_std))
            # print("Time is: " + str("%.2f" % Time) + " with std: " + str("%.2f" % Time_std))

            new_row = [BN, size_of_sampled_dataset, reliability_criterion,
                       Accuracy, Precision, Recall, F1, CI_number, Time,
                       Precision_std, Recall_std, F1_std, CI_number_std, Time_std]
            result.append(new_row)

    # write and save the experiment result as a csv file
    df_for_result = pd.DataFrame(result, columns=COLUMNS)
    result_file_path = f'{working_dir}/results/pc_stable_result_{alpha}_{K}.csv'
    if not os.path.exists(result_file_path):
        df_for_result.to_csv(result_file_path, mode='w')
    else:
        df_for_result.to_csv(result_file_path, mode='a', header=False)
