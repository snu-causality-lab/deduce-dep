# import os
#
# import pandas as pd
#
# from cddd.evaluation import deduce_evaluation
#
#
# def deduce_experiment(BN, working_dir):
#     dataset_sizes = [200, 500, 1000, 2000, 5000]
#     reliability_criterions = ['no']
#     non_sorted_portions = [0]  # hyperparameter for conditional sorting method
#     correction_rules = ['NO']
#
#     # constants
#     sampling_number = 10
#     method = "HITON_PC"
#     h_ps = 5
#     max_k = 3
#     alpha = 0.01
#     isdiscrete = True
#
#     # experiment results
#     columns = ['BN', 'size_of_sampled_dataset', 'correction_rule', 'reliability_criterion', 'non_sorted_portion',
#                'F1', 'Precision', 'Recall', 'Distance']
#     result = []
#
#     # for BN in dataset_BNs:
#     for size_of_sampled_dataset in dataset_sizes:
#         real_graph_path = f"{working_dir}/data/Ground_truth/" + BN + "_true.txt"
#         data_path = f"{working_dir}/data/Sampled_datasets/" + BN + "_" + str(size_of_sampled_dataset) + '_v'
#
#         file_number = sampling_number
#         num_para = pd.read_csv(data_path + '1.csv').shape[1]
#         list_target = [i for i in range(num_para)]
#
#         for correction_rule in correction_rules:
#             for reliability_criterion in reliability_criterions:
#                 if reliability_criterion in ['conditional_sorting']:
#                     for non_sorted_portion in non_sorted_portions:
#                         F1, Precision, Recall, Distance = deduce_evaluation(
#                             method, data_path, num_para, list_target, real_graph_path, isdiscrete, correction_rule, file_number,
#                             alpha, reliability_criterion, h_ps, max_k, non_sorted_portion)
#
#                         print("------------------------------------------------------")
#                         print("the BN of dataset is:", BN)
#                         print("the size of dataset is:", size_of_sampled_dataset)
#                         print("Correction rule is:", correction_rule)
#                         print("Reliability criterion:", reliability_criterion)
#                         print("Non-sorted portion:", non_sorted_portion)
#                         print()
#                         print("F1 is: " + str("%.2f " % F1))
#                         print("Precision is: " + str("%.2f" % Precision))
#                         print("Recall is: " + str("%.2f" % Recall))
#                         print("Distance is: " + str("%.2f" % Distance))
#
#                         new_row = [BN, size_of_sampled_dataset, correction_rule, reliability_criterion, non_sorted_portion,
#                                    F1, Precision, Recall, Distance]
#                         result.append(new_row)
#
#                 else:
#                     non_sorted_portion = 0
#                     F1, Precision, Recall, Distance = deduce_evaluation(
#                         method, data_path, num_para, list_target, real_graph_path, isdiscrete, correction_rule,
#                         file_number,
#                         alpha, reliability_criterion, h_ps, max_k)
#
#                     print("------------------------------------------------------")
#                     print("the BN of dataset is:", BN)
#                     print("the size of dataset is:", size_of_sampled_dataset)
#                     print("Correction rule is:", correction_rule)
#                     print("Reliability criterion:", reliability_criterion)
#                     print("Non-sorted portion:", non_sorted_portion)
#                     print()
#                     print("F1 is: " + str("%.2f " % F1))
#                     print("Precision is: " + str("%.2f" % Precision))
#                     print("Recall is: " + str("%.2f" % Recall))
#                     print("Distance is: " + str("%.2f" % Distance))
#
#                     new_row = [BN, size_of_sampled_dataset, correction_rule, reliability_criterion, non_sorted_portion,
#                                F1, Precision, Recall, Distance]
#                     result.append(new_row)
#
#     # write and save the experiment result as a csv file
#     df_for_result = pd.DataFrame(result, columns=columns)
#     result_file_path = f'{working_dir}/results/cond_result.csv'
#
#     if not os.path.exists(result_file_path):
#         df_for_result.to_csv(result_file_path, mode='w')
#     else:
#         df_for_result.to_csv(result_file_path, mode='a', header=False)
