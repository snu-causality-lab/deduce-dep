from itertools import product

import pandas as pd

from cddd.cit import ci_test_factory
from cddd.evaluation import cond_evaluation
from cddd.utils import safe_save_to_csv


def cond_experiment(BN, ci_tester_name, working_dir, dataset_sizes=(200, 500, 1000, 2000), sampling_number=30):
    reliability_criterions = ['no', 'deductive_reasoning']
    correction_rules = ['AND']
    ci_tester = ci_test_factory(ci_tester_name)

    # constants
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
                    outs = cond_evaluation(data_path, num_para, list_target, real_graph_path, correction_rule, file_number, alpha, reliability_criterion, K=K, ci_tester=ci_tester)
                    result.append([BN, size_of_sampled_dataset, correction_rule, reliability_criterion, *outs])

        result_file_path = f'{working_dir}/results/cond_result_{alpha}_{K}.csv'
        safe_save_to_csv(result, columns, result_file_path)

