import pandas as pd

from cddd.cit import ci_test_factory
from cddd.evaluation import cond_evaluation
from cddd.evaluation import pc_stable_evaluation, complete_pc_stable_evaluation
from cddd.utils import safe_save_to_csv


def performance_experiment(BN, size_of_sampled_dataset, sampling_number, algo, ci_tester_name, alpha, working_dir,
                    reliability_criterions, K, is_orientation):
    """
    Execute performance experiment.

    Args:
        BN: Bayesian network
        size_of_sampled_dataset: size of sampled dataset
        sampling_number: the total number of sampled datasets
        algo: structure learning algorithm
        ci_tester_name: the name of conditional independence test (CIT) to use
        alpha: significance level for CIT
        working_dir: working directory
        reliability_criterions: reliability criterion used in structure learning
        K: recursion threshold parameter for deduce-dep
        is_orientation: whether orientation step in PC algorithm is executed

    Returns:

    """

    ci_tester = ci_test_factory(ci_tester_name)

    columns = ['BN', 'size_of_sampled_dataset', 'reliability_criterion',
               'F1', 'Precision', 'Recall', 'CI_number', 'Time',
               'F1_std', 'Precision_std', 'Recall_std', 'CI_number_std', 'Time_std']

    if algo == 'PC':
        columns = ['BN', 'size_of_sampled_dataset', 'reliability_criterion',
                   'accuracy', 'f1', 'precision', 'recall',
                   'SHD', 'CI_number', 'Time',
                   'accuracy_std', 'f1_std', 'precision_std', 'recall_std',
                   'SHD_std', 'CI_number_std', 'Time_std']

    # experiment results
    result = []
    real_graph_path = f"{working_dir}/data/Ground_truth/{BN}_true.txt"
    data_path = f"{working_dir}/data/Sampled_datasets/{BN}_{size_of_sampled_dataset}_v"
    file_number = sampling_number

    if algo == 'HITON-PC':
        num_para = pd.read_csv(data_path + '1.csv').shape[1]
        list_target = [i for i in range(num_para)]

        for reliability_criterion in reliability_criterions:
            outs = cond_evaluation(data_path, num_para, list_target, real_graph_path, 'AND', file_number, alpha, reliability_criterion, K=K, ci_tester=ci_tester)
            result.append([BN, size_of_sampled_dataset, reliability_criterion, *outs])

    elif algo == 'PC':
        for reliability_criterion in reliability_criterions:
            outs = complete_pc_stable_evaluation(data_path, real_graph_path, file_number, alpha, reliability_criterion,
                                                K=K, ci_tester=ci_tester, is_orientation = is_orientation)
            result.append([BN, size_of_sampled_dataset, reliability_criterion, *outs])

    result_file_path = f'{working_dir}/results/{algo}_result_{alpha}_{K}.csv'
    safe_save_to_csv(result, columns, result_file_path)