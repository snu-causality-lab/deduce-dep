# import os
# import random
# from collections import Counter
# from itertools import combinations
# from itertools import product
#
# import networkx as nx
# import numpy as np
# import pandas as pd
#
# from cddd.cit import cond_indep_test
# from cddd.deductive_reasoning import deduce_dep
# from cddd.evaluation import get_adj_mat
#
#
# def invivo_fn_experiment(BN, working_dir):
#     # experiments settings
#     dataset_sizes = [200, 500, 1000, 2000]
#     sampling_number = 30
#     is_discrete = True
#
#     # alpha = 0.01
#     # K = 2
#     Alphas = [0.01, 0.05]
#     Ks = [0, 1, 2]
#     random.seed(0)
#
#     for alpha, K in product(Alphas, Ks):
#         # experiment results
#         columns = ['BN', 'data_set_size', 'is_deductive_reasoning',
#                    'Accuracy', 'Precision', 'Recall', 'F1']
#         result = []
#
#         for size_of_sampled_dataset in dataset_sizes:
#             real_graph_path = f"{working_dir}/data/Ground_truth/" + BN + "_true.txt"
#             data_path = f"{working_dir}/data/Sampled_datasets/" + BN + "_" + str(size_of_sampled_dataset) + '_v'
#             data = pd.read_csv(data_path + '1.csv')
#             number_of_data, num_vars = np.shape(data)
#
#             true_adj_mat = get_adj_mat(num_vars, real_graph_path)
#             true_graph = nx.DiGraph(true_adj_mat)
#
#             for m in range(sampling_number):
#                 completePath = data_path + str(m + 1) + ".csv"
#                 data = pd.read_csv(completePath)
#                 number_of_data, num_vars = np.shape(data)
#                 data.columns = [i for i in range(num_vars)]
#
#                 size_of_dataset, num_of_variables = np.shape(data)
#                 adj_mat = [[1 if i != j else 0 for j in range(num_of_variables)] for i in range(num_of_variables)]
#                 sepsets = dict()
#                 consets = dict()
#                 add_ci_set = []
#
#                 results_stat = []
#                 results_deduce = []
#                 results_all = []
#
#                 for k in range(num_of_variables - 1):
#                     marker = []
#                     for target in range(num_of_variables):
#                         adj_target = [i for i in range(num_of_variables) if (adj_mat[i][target] == 1) and (i != target)]
#                         for candidate in adj_target:
#                             conditioning_set_pool = list(set(adj_target) - {candidate})
#                             if len(conditioning_set_pool) >= k:
#                                 k_length_conditioning_sets = combinations(conditioning_set_pool, k)
#                                 for cond_set in k_length_conditioning_sets:
#                                     truth = nx.d_separated(true_graph, {target}, {candidate}, set(cond_set))
#                                     pval, _ = cond_indep_test(data, target, candidate, cond_set, is_discrete)
#                                     stat_estim = (pval > alpha)
#
#                                     if pval > alpha:
#                                         deduce_estim = not (deduce_dep(data, target, candidate, cond_set, K, alpha, add_ci_set, sepsets, consets))
#                                     else:
#                                         deduce_estim = False
#
#                                     results_stat.append((truth, stat_estim))
#                                     results_deduce.append((truth, deduce_estim))
#                                     results_all.append((truth, stat_estim, deduce_estim))
#
#                                     if pval > alpha:
#                                         sepsets[tuple(sorted([target, candidate]))] = cond_set
#                                         marker.append([tuple(sorted([target, candidate])), cond_set])
#                                         break
#                                     else:
#                                         consets[tuple(sorted([target, candidate]))] = cond_set
#
#                     for pair_tuple, cond_set in marker:
#                         sepsets[pair_tuple] = cond_set
#                         var1, var2 = pair_tuple
#                         adj_mat[var1][var2] = 0
#                         adj_mat[var2][var1] = 0
#
#                 for i, results in enumerate([results_stat, results_deduce]):
#                     counter = Counter(results)
#                     TP = counter.get((False, False), 0)
#                     TN = counter.get((True, True), 0)
#                     FP = counter.get((True, False), 0)
#                     FN = counter.get((False, True), 0)
#
#                     accuracy = (TP + TN) / (TP + FN + FP + TN)
#                     if (TP + FP) == 0:
#                         precision = 0
#                     else:
#                         precision = TP / (TP + FP)
#                     if (TP + FN) == 0:
#                         recall = 0
#                     else:
#                         recall = TP / (TP + FN)
#                     if (precision + recall) == 0:
#                         f1 = 0
#                     else:
#                         f1 = (2 * precision * recall) / (precision + recall)
#
#                     is_deductive_reasoning = (i != 0)
#
#                     # print("BN:", BN)
#                     # print("size_of_sampled_dataset:", size_of_sampled_dataset)
#                     # print("<results_deduce>" if is_deductive_reasoning else "<results_stat>")
#                     # print("TP, TN, FP, FN:", TP, TN, FP, FN)
#                     # print("accuracy:", accuracy)
#                     # print("precision:", precision)
#                     # print("recall:", recall)
#                     # print("f1:", f1)
#                     # print()
#                     new_row = [BN, size_of_sampled_dataset, is_deductive_reasoning,
#                                accuracy, precision, recall, f1]
#                     result.append(new_row)
#
#         # write and save the experiment result as a csv file
#         df_for_result = pd.DataFrame(result, columns=columns)
#         result_file_path = f'{working_dir}/results/invivo_fn_result_{alpha}_{K}.csv'
#         if not os.path.exists(result_file_path):
#             df_for_result.to_csv(result_file_path, mode='w')
#         else:
#             df_for_result.to_csv(result_file_path, mode='a', header=False)
#
# # if __name__ == '__main__':
# #     from joblib import Parallel, delayed
# #     from itertools import product
# #     import multiprocessing as mp
# #
# #     BNs = ['ER_10_12', 'ER_10_15', 'ER_10_20', 'ER_20_24', 'ER_20_30', 'ER_20_40', 'ER_30_36', 'ER_30_45', 'ER_30_60']
# #     BNs += ['alarm', 'asia', 'child', 'insurance', 'sachs', 'water']
# #
# #     Parallel(n_jobs=mp.cpu_count())(delayed(invivo_fn_experiment)(BN) for BN in BNs)
