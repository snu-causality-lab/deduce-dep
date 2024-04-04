from itertools import combinations, product

import networkx as nx


def estimate_cpdag(skel_graph, sep_set):
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

    # directly modifies the dag
    apply_meeks_rule(dag)

    return dag


def is_undirected(dag, i, j):  # "_has_both_edges" is about syntax, is_undirected is about semantic
    return dag.has_edge(i, j) and dag.has_edge(j, i)


def is_adjacent(dag, i, j):
    return dag.has_edge(i, j) or dag.has_edge(j, i)


def apply_meeks_rule(dag: nx.DiGraph):
    # For all the combination of nodes i and j, apply the following
    # rules.
    # old_dag = dag.copy()
    # node_ids = old_dag.nodes()
    changed = True
    while changed:
        changed = False
        for _i, _j in list(dag.edges()):  # because dag.edges are changing, wrap with list
            for i, j in ((_i, _j), (_j, _i)):
                if not is_undirected(dag, i, j):
                    continue

                # Rule 1: Orient i-j into i->j whenever there is an arrow k->i
                # such that k and j are nonadjacent.
                #
                # Check if i-j.
                if is_undirected(dag, i, j):
                    # Look all the predecessors of i.
                    for k in dag.predecessors(i):
                        # Skip if there is an arrow i->k.
                        if dag.has_edge(i, k):
                            continue
                        # Skip if k and j are adjacent.
                        if is_adjacent(dag, k, j):
                            continue
                        # Make i-j into i->j
                        # _logger.debug('R1: remove edge (%s, %s)' % (j, i))
                        dag.remove_edge(j, i)
                        changed = True
                        break

                # Rule 2: Orient i-j into i->j whenever there is a chain
                # i->k->j.
                #
                # Check if i-j.
                if is_undirected(dag, i, j):
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
                        changed = True

                # Rule 3: Orient i-j into i->j whenever there are two chains
                # i-k->j and i-l->j such that k and l are nonadjacent.
                #
                # Check if i-j.
                if is_undirected(dag, i, j):
                    # Find nodes k where i-k.
                    adj_i = set()
                    for k in dag.successors(i):
                        if dag.has_edge(k, i):
                            adj_i.add(k)
                    # For all the pairs of nodes in adj_i,
                    for (k, l) in combinations(adj_i, 2):
                        # Skip if k and l are adjacent.
                        if is_adjacent(dag, k, l):
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
                        changed = True
                        break

                # Rule 4: Orient i-j into i->j whenever there are two chains
                # i-k->l and k->l->j such that k and j are nonadjacent and i and l are adjacent.

                if is_undirected(dag, i, j):
                    # Find nodes k where i-k.
                    adj_i = set()
                    for k in dag.successors(i):
                        if dag.has_edge(k, i):
                            adj_i.add(k)

                    # Find nodes l where l->j
                    pa_j = set()
                    for l in dag.predecessors(j):
                        if not dag.has_edge(j, l):
                            pa_j.add(l)

                    # For all pairs of (k, l),
                    for k, l in product(adj_i, pa_j):
                        # Check k -> l
                        if dag.has_edge(k, l) and not dag.has_edge(l, k):
                            # check adjacency between i and l & non-adjacency between k and j
                            if is_adjacent(dag, i, l) and not is_adjacent(dag, k, j):
                                dag.remove_edge(j, i)
                                changed = True
                                break

        # if nx.is_isomorphic(dag, old_dag):
        #     break
        if not changed:
            break
        # old_dag = dag.copy()
