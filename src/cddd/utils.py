import os
from typing import List
from itertools import combinations

import networkx as nx
import pandas as pd
from filelock import FileLock

def DAG_to_CPDAG(adj_mat):
    GT = nx.DiGraph(adj_mat)
    nodes = list(GT.nodes)
    rtn_adj_mat = adj_mat + adj_mat.T

    for a, b in combinations(nodes, r=2):
        if (not rtn_adj_mat[a][b]) and (not rtn_adj_mat[b][a]):
            common_child = set(GT.successors(a)) & set(GT.successors(b))
            for child in common_child:
                rtn_adj_mat[a][child] = 1
                rtn_adj_mat[child][a] = 0
                rtn_adj_mat[b][child] = 1
                rtn_adj_mat[child][b] = 0
    return rtn_adj_mat


def safe_save_to_csv(result: List[List], columns: List[str], file_path: str):
    df = pd.DataFrame(result, columns=columns)
    lock = FileLock(file_path + '.lock')
    with lock:
        if not os.path.exists(file_path):
            df.to_csv(file_path, mode='w')
        else:
            df.to_csv(file_path, mode='a', header=False)
