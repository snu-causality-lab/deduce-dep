import collections
import time
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from cddd.algos.HITON_PC import HITON_PC
from cddd.algos.HITON_PC_with_CI_ORACLE import HITON_PC_oracle
from cddd.algos.PC_STABLE import pc_stable
from cddd.algos.PC_STABLE_with_CI_ORACLE import pc_stable_oracle
from cddd.correction import correction
from cddd.utils import DAG_to_CPDAG


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



def get_SHD(oracle_adj_mat, estim_adj_mat):
    # DAG -> CPDAG
    diff = np.abs(oracle_adj_mat - estim_adj_mat)
    diff = diff + diff.transpose()
    diff[diff > 1] = 1
    return np.sum(diff) / 2


def HITON_PC_evaluation(path, all_number_Para, target_list, real_graph_path, rule, filenumber=10, alaph=0.01,
                        reliability_criterion='classic', K=1, ci_tester=None):
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

        for i, target in enumerate(target_list):
            start_time = time.time()
            PC, _, ci_number = HITON_PC(data, assoc, target, alaph, reliability_criterion, K, ci_tester=ci_tester)
            end_time = time.time()
            time_lapsed = end_time - start_time

            total_ci_number += ci_number
            CI_numbers.append(ci_number)
            total_time += time_lapsed
            Times.append(time_lapsed)
            ResPC[i] = PC

        correction(ResPC, rule)
        Distance, F1, Precision, Recall = local_metric_evaluation(Distance, F1, Precision, Recall, ResPC, OraclePC,
                                                                  target_list)

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

    return F1 / commonDivisor, Precision / commonDivisor, Recall / commonDivisor, total_ci_number / commonDivisor, total_time / commonDivisor, \
        F1s_std, Precisions_std, Recalls_std, CI_numbers_std, Times_std


def PC_stable_evaluation(path, real_graph_path, filenumber=10, alpha=0.01, reliability_criterion='classic',
                         K=1, ci_tester=None, is_orientation = False):
    # pre_set variables are zero
    adj_f1s = []
    adj_precisions = []
    adj_recalls = []

    SHDs = []
    CI_numbers = []
    Times = []

    oracle_adj_mat = get_oracle_adj_mat(path, real_graph_path, is_orientation=is_orientation)
    for m in range(filenumber):
        data = load_data(path, m)

        start_time = time.time()
        estim_adj_mat, sepsets, ci_number = pc_stable(data, alpha, reliability_criterion=reliability_criterion,
                                                      is_orientation=is_orientation, K=K, ci_tester=ci_tester)
        end_time = time.time()
        time_lapsed = end_time - start_time

        _, adj_precision, adj_recall, adj_f1 = global_skeleton_metric_evaluation(oracle_adj_mat, estim_adj_mat)
        SHD = get_SHD(oracle_adj_mat, estim_adj_mat)

        adj_precisions.append(adj_precision)
        adj_recalls.append(adj_recall)
        adj_f1s.append(adj_f1)

        SHDs.append(SHD)
        CI_numbers.append(ci_number)
        Times.append(time_lapsed)

    adj_f1s_mean, adj_f1s_std = np.mean(adj_f1s), np.std(adj_f1s)
    adj_precisions_mean, adj_precisions_std = np.mean(adj_precisions), np.std(adj_precisions)
    adj_recalls_mean, adj_recalls_std = np.mean(adj_recalls), np.std(adj_recalls)

    SHDs_mean, SHDs_std = np.mean(SHDs), np.std(SHDs)
    CI_numbers_mean, CI_numbers_std = np.mean(CI_numbers), np.std(CI_numbers)
    Times_mean, Times_std = np.mean(Times), np.std(Times)

    return adj_f1s_mean, adj_precisions_mean, adj_recalls_mean, \
        SHDs_mean, CI_numbers_mean, Times_mean, \
        adj_f1s_std, adj_precisions_std, adj_recalls_std, \
        SHDs_std, CI_numbers_std, Times_std


def get_oracle_adj_mat(path, real_graph_path, is_orientation = True):
    examplePath = path + str(1) + ".csv"
    _data = pd.read_csv(examplePath)
    number, all_number_Para = np.shape(_data)
    true_adj_mat = get_adj_mat(all_number_Para, real_graph_path)
    true_graph = nx.DiGraph(true_adj_mat)
    oracle_adj_mat = pc_stable_oracle(true_adj_mat, true_graph, is_orientation=is_orientation)
    return oracle_adj_mat


def load_data(path, m):
    completePath = path + str(m + 1) + ".csv"
    data = pd.read_csv(completePath)
    number, kVar = np.shape(data)
    data.columns = [i for i in range(kVar)]
    return data


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
