import os
from itertools import product

import pandas as pd

from cddd.evaluation import cond_evaluation


def cond_experiment(BN, working_dir, dataset_sizes=(200, 500, 1000, 2000), sampling_number=30):
    reliability_criterions = ['no', 'deductive_reasoning']
    correction_rules = ['AND']

    # constants
    isdiscrete = True
    Ks = [0, 1, 2]
    Alphas = [0.05, 0.01]

    columns = ['BN', 'size_of_sampled_dataset', 'correction_rule', 'reliability_criterion',
               'F1', 'Precision', 'Recall', 'Distance', 'CI_number', 'Time',
               'F1_std', 'Precision_std', 'Recall_std', 'CI_number_std', 'Time_std']

    for K, alpha in product(Ks, Alphas):
        # experiment results

        result = []

        # for BN in dataset_BNs:
        for size_of_sampled_dataset in dataset_sizes:
            real_graph_path = f"{working_dir}/data/Ground_truth/{BN}_true.txt"
            data_path = f"{working_dir}/data/Sampled_datasets/{BN}_{size_of_sampled_dataset}_v"

            file_number = sampling_number
            num_para = pd.read_csv(data_path + '1.csv').shape[1]
            list_target = [i for i in range(num_para)]

            for correction_rule in correction_rules:
                for reliability_criterion in reliability_criterions:
                    F1, Precision, Recall, Distance, CI_number, Time, \
                        F1_std, Precision_std, Recall_std, CI_number_std, Time_std = cond_evaluation(data_path, num_para, list_target, real_graph_path, isdiscrete, correction_rule, file_number, alpha, reliability_criterion, K=K)

                    # print("------------------------------------------------------")
                    # print("the BN of dataset is:", BN)
                    # print("the size of dataset is:", size_of_sampled_dataset)
                    # print("Correction rule is:", correction_rule)
                    # print("Reliability criterion:", reliability_criterion)
                    # print()
                    # print("F1 is: " + str("%.2f " % F1) + " with std: " + str("%.2f" % F1_std))
                    # print("Precision is: " + str("%.2f" % Precision) + " with std: " + str("%.2f" % Precision_std))
                    # print("Recall is: " + str("%.2f" % Recall) + " with std: " + str("%.2f" % Recall_std))
                    # # print("Distance is: " + str("%.2f" % Distance))
                    # print("CI_number is: " + str("%.2f" % CI_number) + " with std: " + str("%.2f" % CI_number_std))
                    # print("Time is: " + str("%.2f" % Time) + " with std: " + str("%.2f" % Time_std))

                    new_row = [BN, size_of_sampled_dataset, correction_rule, reliability_criterion,
                               F1, Precision, Recall, Distance, CI_number, Time,
                               F1_std, Precision_std, Recall_std, CI_number_std, Time_std]
                    result.append(new_row)

        # write and save the experiment result as a csv file
        df_for_result = pd.DataFrame(result, columns=columns)
        result_file_path = f'{working_dir}/results/cond_result_{alpha}_{K}.csv'

        if not os.path.exists(result_file_path):
            df_for_result.to_csv(result_file_path, mode='w')
        else:
            df_for_result.to_csv(result_file_path, mode='a', header=False)
