from cddd.cit import ci_test_factory
from cddd.evaluation import pc_stable_evaluation
from cddd.utils import safe_save_to_csv


def pc_stable_experiment(BN, ci_tester_name, working_dir, size_of_sampled_dataset, sampling_number, K, alpha):
    reliability_criterions = ['no', 'deductive_reasoning']
    # constants

    ci_tester = ci_test_factory(ci_tester_name)
    COLUMNS = ['BN', 'size_of_sampled_dataset', 'reliability_criterion',
               'Accuracy', 'Precision', 'Recall', 'F1', 'CI_number', 'Time',
               'Precision_std', 'Recall_std', 'F1_std', 'CI_number_std', 'Time_std']

    # experiment results
    result = []
    real_graph_path = f"{working_dir}/data/Ground_truth/" + BN + "_true.txt"
    data_path = f"{working_dir}/data/Sampled_datasets/" + BN + "_" + str(size_of_sampled_dataset) + '_v'

    file_number = sampling_number

    for reliability_criterion in reliability_criterions:
        outs = pc_stable_evaluation(data_path, real_graph_path, file_number, alpha, reliability_criterion, K=K, ci_tester=ci_tester)

        result.append([BN, size_of_sampled_dataset, reliability_criterion, *outs])

    result_file_path = f'{working_dir}/results/pc_stable_result_{alpha}_{K}.csv'
    safe_save_to_csv(result, COLUMNS, result_file_path)
    # print(f'pc_stable_experiment {(BN, ci_tester_name, working_dir, size_of_sampled_dataset, sampling_number, K, alpha)}')
