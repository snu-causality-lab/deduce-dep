from itertools import combinations, permutations
import numpy as np
import networkx as nx


def pc_stable_oracle(true_adj_mat, true_graph, is_orientation = False):
    num_of_variables = len(true_adj_mat)
    adj_mat = [[1 if i != j else 0 for j in range(num_of_variables)] for i in range(num_of_variables)]
    sepsets = dict()

    for k in range(num_of_variables - 1):
        marker = []
        for target in range(num_of_variables):
            adj_target = [i for i in range(num_of_variables) if ((tuple(sorted([target, i])) not in sepsets) and (i != target))]
            for candidate in adj_target:
                conditioning_set_pool = list(set(adj_target) - {candidate})
                if len(conditioning_set_pool) >= k:
                    k_length_conditioning_sets = combinations(conditioning_set_pool, k)
                    for cond_set in k_length_conditioning_sets:
                        ci_oracle = nx.d_separated(true_graph, {target}, {candidate}, set(cond_set))
                        if ci_oracle:
                            marker.append([tuple(sorted([target, candidate])), cond_set])
                            break

        for pair_tuple, cond_set in marker:
            sepsets[pair_tuple] = cond_set
            var1, var2 = pair_tuple
            adj_mat[var1][var2] = 0
            adj_mat[var2][var1] = 0

    if not is_orientation:
        return adj_mat
    else:
        adj_mat = np.array(adj_mat)
        skel_graph = nx.DiGraph(adj_mat)
        dag = estimate_oracle_cpdag(skel_graph, sepsets)
        adj_mat = nx.to_numpy_array(dag)
        return adj_mat

def estimate_oracle_cpdag(skel_graph, sep_set):
    """Estimate a CPDAG from the skeleton graph and separation sets
    returned by the estimate_skeleton() function.

    Args:
        skel_graph: A skeleton graph (an undirected networkx.Graph).
        sep_set: An 2D-array of separation set.
            The contents look like something like below.
                sep_set[i][j] = set([k, l, m])

    Returns:
        An estimated DAG.
    """
    # dag = skel_graph.to_directed()
    dag = skel_graph
    node_ids = skel_graph.nodes()
    for (i, j) in combinations(node_ids, 2):
        adj_i = set(dag.successors(i))
        if j in adj_i:
            continue
        adj_j = set(dag.successors(j))
        if i in adj_j:
            continue
        # if sep_set[i][j] is None:
        if not sep_set[tuple(sorted([i, j]))]:
            continue
        common_k = adj_i & adj_j
        for k in common_k:
            if k not in sep_set[tuple(sorted([i, j]))]:
                if dag.has_edge(k, i):
                    # _logger.debug('S: remove edge (%s, %s)' % (k, i))
                    dag.remove_edge(k, i)
                if dag.has_edge(k, j):
                    # _logger.debug('S: remove edge (%s, %s)' % (k, j))
                    dag.remove_edge(k, j)

    def _has_both_edges(dag, i, j):
        return dag.has_edge(i, j) and dag.has_edge(j, i)

    def _has_any_edge(dag, i, j):
        return dag.has_edge(i, j) or dag.has_edge(j, i)

    def _has_one_edge(dag, i, j):
        return ((dag.has_edge(i, j) and (not dag.has_edge(j, i))) or
                (not dag.has_edge(i, j)) and dag.has_edge(j, i))

    def _has_no_edge(dag, i, j):
        return (not dag.has_edge(i, j)) and (not dag.has_edge(j, i))

    # For all the combination of nodes i and j, apply the following
    # rules.
    old_dag = dag.copy()
    while True:
        for (i, j) in permutations(node_ids, 2):
            # Rule 1: Orient i-j into i->j whenever there is an arrow k->i
            # such that k and j are nonadjacent.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Look all the predecessors of i.
                for k in dag.predecessors(i):
                    # Skip if there is an arrow i->k.
                    if dag.has_edge(i, k):
                        continue
                    # Skip if k and j are adjacent.
                    if _has_any_edge(dag, k, j):
                        continue
                    # Make i-j into i->j
                    # _logger.debug('R1: remove edge (%s, %s)' % (j, i))
                    dag.remove_edge(j, i)
                    break

            # Rule 2: Orient i-j into i->j whenever there is a chain
            # i->k->j.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Find nodes k where k is i->k.
                succs_i = set()
                for k in dag.successors(i):
                    if not dag.has_edge(k, i):
                        succs_i.add(k)
                # Find nodes j where j is k->j.
                preds_j = set()
                for k in dag.predecessors(j):
                    if not dag.has_edge(j, k):
                        preds_j.add(k)
                # Check if there is any node k where i->k->j.
                if len(succs_i & preds_j) > 0:
                    # Make i-j into i->j
                    # _logger.debug('R2: remove edge (%s, %s)' % (j, i))
                    dag.remove_edge(j, i)

            # Rule 3: Orient i-j into i->j whenever there are two chains
            # i-k->j and i-l->j such that k and l are nonadjacent.
            #
            # Check if i-j.
            if _has_both_edges(dag, i, j):
                # Find nodes k where i-k.
                adj_i = set()
                for k in dag.successors(i):
                    if dag.has_edge(k, i):
                        adj_i.add(k)
                # For all the pairs of nodes in adj_i,
                for (k, l) in combinations(adj_i, 2):
                    # Skip if k and l are adjacent.
                    if _has_any_edge(dag, k, l):
                        continue
                    # Skip if not k->j.
                    if dag.has_edge(j, k) or (not dag.has_edge(k, j)):
                        continue
                    # Skip if not l->j.
                    if dag.has_edge(j, l) or (not dag.has_edge(l, j)):
                        continue
                    # Make i-j into i->j.
                    # _logger.debug('R3: remove edge (%s, %s)' % (j, i))
                    dag.remove_edge(j, i)
                    break

            # Rule 4: Orient i-j into i->j whenever there are two chains
            # i-k->l and k->l->j such that k and j are nonadjacent.
            #
            # However, this rule is not necessary when the PC-algorithm
            # is used to estimate a DAG.

        if nx.is_isomorphic(dag, old_dag):
            break
        old_dag = dag.copy()

    return dag
