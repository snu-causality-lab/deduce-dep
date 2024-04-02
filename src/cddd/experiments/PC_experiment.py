import os

import pandas as pd

from cddd.evaluation import pc_evaluation


def pc_experiment(BN, working_dir, dataset_sizes=(200, 500, 1000, 2000), sampling_number=30, alphas=(0.01, 0.05)):
    for alpha in alphas:
        _pc_experiment_core(BN, working_dir, dataset_sizes, sampling_number, alpha)


def _pc_experiment_core(BN, working_dir, dataset_sizes=(200, 500, 1000, 2000), sampling_number=30, alpha=0.05):
    assert alpha == 0.01 or alpha == 0.05
    alpha_str = {0.01: '001', 0.05: '005'}
    reliability_criterions = ['no', 'deductive_reasoning']
    # constants
    isdiscrete = True

    # experiment results
    columns = ['BN', 'size_of_sampled_dataset', 'reliability_criterion',
               'Accuracy', 'Precision', 'Recall', 'F1', 'CI_number', 'Time',
               'Precision_std', 'Recall_std', 'F1_std', 'CI_number_std', 'Time_std']
    result = []

    for size_of_sampled_dataset in dataset_sizes:
        real_graph_path = f"{working_dir}/data/Ground_truth/{BN}_true.txt"
        data_path = f"{working_dir}/data/Sampled_datasets/{BN}_{size_of_sampled_dataset}_v"

        file_number = sampling_number

        for reliability_criterion in reliability_criterions:
            Accuracy, Precision, Recall, F1, CI_number, Time, \
                Precision_std, Recall_std, F1_std, CI_number_std, Time_std = pc_evaluation(data_path, real_graph_path, isdiscrete, file_number, alpha, reliability_criterion)

            new_row = [BN, size_of_sampled_dataset, reliability_criterion,
                       Accuracy, Precision, Recall, F1, CI_number, Time,
                       Precision_std, Recall_std, F1_std, CI_number_std, Time_std]
            result.append(new_row)

    # write and save the experiment result as a csv file
    df_for_result = pd.DataFrame(result, columns=columns)
    result_file_path = f'{working_dir}/results/pc_result_{sampling_number}_alpha_{alpha_str[alpha]}.csv'

    if not os.path.exists(result_file_path):
        df_for_result.to_csv(result_file_path, mode='w')
    else:
        df_for_result.to_csv(result_file_path, mode='a', header=False)
