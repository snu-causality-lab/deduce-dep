import collections
import time

import networkx as nx
import numpy as np
import pandas as pd

from cddd.algos.HITON_PC import HITON_PC
from cddd.algos.HITON_PC_with_CI_ORACLE import HITON_PC_oracle
from cddd.algos.PC import pc
from cddd.algos.PC_STABLE import pc_stable
from cddd.algos.PC_STABLE_with_CI_ORACLE import pc_stable_oracle
from cddd.algos.PC_with_CI_ORACLE import pc_oracle
from cddd.correction import correction


#
# def realMB(kVar, path):
#     graph = np.zeros((kVar, kVar))
#     parents = [[] for _ in range(kVar)]
#     children = [[] for _ in range(kVar)]
#     MB = [[] for _ in range(kVar)]
#     PC = [[] for _ in range(kVar)]
#     spouses = [[] for _ in range(kVar)]
#
#     i = 0
#     with open(path) as fileobject:
#         for line in fileobject:
#             a = line.split(" ")
#             j = 0
#             for n in a:
#                 graph[i, j] = n
#                 j += 1
#             i += 1
#
#     for m in range(kVar):
#         parents[m] = [i for i in range(kVar) if graph[i][m] == 1]
#         children[m] = [i for i in range(kVar) if graph[m][i] == 1]
#
#         PC[m] = list(set(parents[m]).union(set(children[m])))
#
#     for m in range(kVar):
#         for child in children[m]:
#             spouse = parents[int(child)]
#             spouses[m] = list(set(spouses[m]).union(set(spouse)))
#         if m in spouses[m]:
#             spouses[m].remove(m)
#
#     for m in range(kVar):
#         MB[m] = list(set(PC[m]).union(set(spouses[m])))
#
#     return MB, PC, graph


def local_metric_evaluation(Distance, F1, Precision, Recall, ResPC, realpc, target_list):
    for n, target in enumerate(target_list):
        true_positive = list(
            set(realpc[target]).intersection(set(ResPC[n])))
        length_true_positive = len(true_positive)
        length_RealPC = len(realpc[target])
        length_ResPC = len(ResPC[n])
        if length_RealPC == 0:
            if length_ResPC == 0:
                precision = 1
                recall = 1
                distance = 0
                F1 += 1
            else:
                F1 += 0
                precision = 0
                distance = 2 ** 0.5
                recall = 0
        else:
            if length_ResPC != 0:
                precision = length_true_positive / length_ResPC
                recall = length_true_positive / length_RealPC
                distance = ((1 - precision) ** 2 + (1 - recall) ** 2) ** 0.5
                if precision + recall != 0:
                    F1 += 2 * precision * recall / (precision + recall)
            else:
                F1 += 0
                precision = 0
                recall = 0
                distance = 2 ** 0.5
        Distance += distance
        Precision += precision
        Recall += recall
    return Distance, F1, Precision, Recall


def global_skeleton_metric_evaluation(true_adj_mat, estim_adj_mat):
    num_vars = len(true_adj_mat[0])
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for var1 in range(num_vars):
        for var2 in range(var1 + 1, num_vars):  # range(var1+1, num_vars)
            truth = (true_adj_mat[var1][var2] or true_adj_mat[var2][var1])
            estim = (estim_adj_mat[var1][var2] or estim_adj_mat[var2][var1])
            if truth and estim:
                TP += 1
            elif truth and not estim:
                FN += 1
            elif not truth and estim:
                FP += 1
            elif not truth and not estim:
                TN += 1

    accuracy = (TP + TN) / (TP + FN + FP + TN)
    precision = (TP / (TP + FP)) if TP + FP > 0 else 0
    recall = (TP / (TP + FN)) if TP + FN > 0 else 0
    f1 = ((2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0)

    return accuracy, precision, recall, f1


def cond_evaluation(path, all_number_Para, target_list, real_graph_path, rule, filenumber=10, alaph=0.01, reliability_criterion='classic', K=1, ci_tester=None):
    # pre_set variables are zero
    Precision = 0
    Recall = 0
    F1 = 0
    Distance = 0

    length_targets = len(target_list)
    total_ci_number = 0
    total_time = 0

    true_adj_mat = get_adj_mat(all_number_Para, real_graph_path)
    true_graph = nx.DiGraph(true_adj_mat)

    F1s = []
    Precisions = []
    Recalls = []
    CI_numbers = []
    Times = []

    for m in range(filenumber):
        completePath = path + str(m + 1) + ".csv"
        data = pd.read_csv(completePath)
        number, kVar = np.shape(data)
        data.columns = [i for i in range(kVar)]

        ResPC = [[]] * length_targets
        OraclePC = [[]] * length_targets
        assoc = [[0] * kVar for _ in range(kVar)]

        for i, target in enumerate(target_list):
            Oraclepc = HITON_PC_oracle(data, assoc, target, alaph, true_graph, ci_tester=ci_tester)
            OraclePC[i] = Oraclepc

        sepsets = None
        for i, target in enumerate(target_list):
            start_time = time.time()
            PC, sepsets, ci_number = HITON_PC(data, assoc, target, alaph, reliability_criterion, K, ci_tester=ci_tester)
            end_time = time.time()
            time_lapsed = end_time - start_time

            total_ci_number += ci_number
            CI_numbers.append(ci_number)
            total_time += time_lapsed
            Times.append(time_lapsed)
            ResPC[i] = PC

        correction(ResPC, rule)
        Distance, F1, Precision, Recall = local_metric_evaluation(Distance, F1, Precision, Recall, ResPC, OraclePC, target_list)

        F1s.append(F1 / (length_targets * (m + 1)))
        Precisions.append(Precision / (length_targets * (m + 1)))
        Recalls.append(Recall / (length_targets * (m + 1)))

    commonDivisor = length_targets * filenumber

    F1s = np.array(F1s)
    Precisions = np.array(Precisions)
    Recalls = np.array(Recalls)
    CI_numbers = np.array(CI_numbers)
    Times = np.array(Times)

    F1s_std = np.std(F1s)
    Precisions_std = np.std(Precisions)
    Recalls_std = np.std(Recalls)
    CI_numbers_std = np.std(CI_numbers)
    Times_std = np.std(Times)

    return F1 / commonDivisor, Precision / commonDivisor, Recall / commonDivisor, Distance / commonDivisor, total_ci_number / commonDivisor, total_time / commonDivisor, \
        F1s_std, Precisions_std, Recalls_std, CI_numbers_std, Times_std


def pc_evaluation(path, real_graph_path, filenumber, alpha, reliability_criterion='classic', ci_tester=None):
    # pre_set variables are zero
    Accuracy = 0
    Precision = 0
    Recall = 0
    F1 = 0
    total_ci_number = 0
    total_time = 0

    F1s = []
    Precisions = []
    Recalls = []
    CI_numbers = []
    Times = []

    true_adj_dict = get_adj_dict(real_graph_path)
    true_graph = nx.DiGraph(true_adj_dict)
    oracle_adj_mat = pc_oracle(true_adj_dict, true_graph)

    for m in range(filenumber):
        completePath = path + str(m + 1) + ".csv"
        data = pd.read_csv(completePath)
        number, kVar = np.shape(data)
        data.columns = [i for i in range(kVar)]

        start_time = time.time()
        estim_adj_mat, sepsets, ci_number = pc(data, alpha, reliability_criterion, ci_tester=ci_tester)
        end_time = time.time()
        time_lapsed = end_time - start_time
        accuracy, precision, recall, f1 = global_skeleton_metric_evaluation(oracle_adj_mat, estim_adj_mat)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        CI_numbers.append(ci_number)
        Times.append(time_lapsed)

        total_ci_number += ci_number
        total_time += time_lapsed
        Accuracy += accuracy
        Precision += precision
        Recall += recall
        F1 += f1

    commonDivisor = filenumber

    F1s = np.array(F1s)
    Precisions = np.array(Precisions)
    Recalls = np.array(Recalls)
    CI_numbers = np.array(CI_numbers)
    Times = np.array(Times)

    F1s_std = np.std(F1s)
    Precisions_std = np.std(Precisions)
    Recalls_std = np.std(Recalls)
    CI_numbers_std = np.std(CI_numbers)
    Times_std = np.std(Times)

    return Accuracy / commonDivisor, Precision / commonDivisor, Recall / commonDivisor, F1 / commonDivisor, total_ci_number / commonDivisor, total_time / commonDivisor, \
        Precisions_std, Recalls_std, F1s_std, CI_numbers_std, Times_std


def pc_stable_evaluation(path, real_graph_path, filenumber=10, alpha=0.01, reliability_criterion='classic', K=1, ci_tester=None):
    # pre_set variables are zero
    Accuracy = 0
    Precision = 0
    Recall = 0
    F1 = 0
    total_ci_number = 0
    total_time = 0

    F1s = []
    Precisions = []
    Recalls = []
    CI_numbers = []
    Times = []

    examplePath = path + str(1) + ".csv"
    data = pd.read_csv(examplePath)
    number, all_number_Para = np.shape(data)

    true_adj_mat = get_adj_mat(all_number_Para, real_graph_path)
    true_graph = nx.DiGraph(true_adj_mat)
    oracle_adj_mat = pc_stable_oracle(true_adj_mat, true_graph)

    for m in range(filenumber):
        completePath = path + str(m + 1) + ".csv"
        data = pd.read_csv(completePath)
        number, kVar = np.shape(data)
        data.columns = [i for i in range(kVar)]

        start_time = time.time()
        estim_adj_mat, sepsets, ci_number = pc_stable(data, alpha, reliability_criterion, K, ci_tester=ci_tester)
        end_time = time.time()
        time_lapsed = end_time - start_time
        accuracy, precision, recall, f1 = global_skeleton_metric_evaluation(oracle_adj_mat, estim_adj_mat)

        F1s.append(f1)
        Precisions.append(precision)
        Recalls.append(recall)
        CI_numbers.append(ci_number)
        Times.append(time_lapsed)

        total_ci_number += ci_number
        total_time += time_lapsed
        Accuracy += accuracy
        Precision += precision
        Recall += recall
        F1 += f1

    commonDivisor = filenumber

    F1s = np.array(F1s)
    Precisions = np.array(Precisions)
    Recalls = np.array(Recalls)
    CI_numbers = np.array(CI_numbers)
    Times = np.array(Times)

    F1s_std = np.std(F1s)
    Precisions_std = np.std(Precisions)
    Recalls_std = np.std(Recalls)
    CI_numbers_std = np.std(CI_numbers)
    Times_std = np.std(Times)

    return Accuracy / commonDivisor, Precision / commonDivisor, Recall / commonDivisor, F1 / commonDivisor, total_ci_number / commonDivisor, total_time / commonDivisor, \
        Precisions_std, Recalls_std, F1s_std, CI_numbers_std, Times_std


def get_adj_dict(real_graph_path):
    adj_dict = collections.defaultdict(list)
    with open(real_graph_path) as fileobject:
        i = 0
        for line in fileobject:
            a = line.split(" ")
            for j, n in enumerate(a):
                if int(n) == 1:
                    adj_dict[i].append(j)
            i += 1
    return adj_dict


def get_adj_mat(num_vars, real_graph_path):
    i = 0
    with open(real_graph_path) as fileobject:
        true_adj_mat = [['*'] * num_vars for _ in range(num_vars)]
        for line in fileobject:
            a = line.split(" ")
            j = 0
            for n in a:
                true_adj_mat[i][j] = int(n)
                j += 1
            i += 1
    labels = [i for i in range(num_vars)]
    true_adj_mat = pd.DataFrame(true_adj_mat, index=labels, columns=labels)
    return true_adj_mat
