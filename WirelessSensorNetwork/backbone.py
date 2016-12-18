"""Use graph coloring to identify optimal backbones."""

from collections import defaultdict, deque
import heapq
import itertools

import networkx as nx

from utils import time_execution


@time_execution
def smallest_last_ordering(G, with_stats=False):
    """Code I contributed to networkx."""
    H = G.copy(with_data=False)
    ordering = deque()
    if with_stats:
        degree_pairs = deque()
        terminal_clique_size = None

    # Build initial degree list (i.e. the bucket queue data structure)
    degrees = defaultdict(set)
    lbound = float('inf')
    for node, d in H.degree():
        degrees[d].add(node)
        lbound = min(lbound, d)  # Lower bound on min-degree.

    def find_min_degree():
        # Save time by starting the iterator at `lbound`, not 0.
        # The value that we find will be our new `lbound`, which we set later.
        return next(d for d in itertools.count(lbound) if d in degrees)

    for _ in G:
        # Pop a min-degree node and add it to the list.
        min_degree = find_min_degree()
        u = degrees[min_degree].pop()
        if not degrees[min_degree]:  # Clean up the degree list.
            del degrees[min_degree]
        ordering.appendleft(u)

        # Update degrees of removed node's neighbors.
        for v in H[u]:
            degree = H.degree(v)
            degrees[degree].remove(v)
            if not degrees[degree]:  # Clean up the degree list.
                del degrees[degree]
            degrees[degree - 1].add(v)

        if with_stats:
            degree_pairs.appendleft((H.degree(u), G.degree(u)))
            if terminal_clique_size is None and len(degrees) == 1:
                terminal_clique_size = min(d for _, d in H.degree()) + 1

        # Finally, remove the node.
        H.remove_node(u)
        lbound = min_degree - 1  # Subtract 1 in case of tied neighbors.

    if with_stats:
        ordering = (ordering, degree_pairs, terminal_clique_size)
    return ordering


@time_execution
def greedy_color(G, nodes=None):
    colors = {}
    if G:
        if nodes is None:
            nodes = smallest_last_ordering(G)

        for u in nodes:
            # set to track neighbors' colors
            neighbor_colors = set()
            for v in G[u]:
                if v in colors:
                    neighbor_colors.add(colors[v])
            for color in itertools.count():
                if color not in neighbor_colors:
                    break
            colors[u] = color

    return colors


def independent_sets_from_colors(colors):
    lst = [set() for _ in range(1 + max(colors.values()))]
    for node, color in colors.items():
        lst[color].add(node)
    return lst


def count_faces(G):
    # F + V = E + C + 1 where C is the number of components.
    # So F = E - V + C + 1
    E = nx.number_of_edges(G)
    V = nx.number_of_nodes(G)
    C = nx.number_connected_components(G)
    return E - V + C + 1


def select_backbones(G, indept_sets):
    """Compute which pair of indep't sets makes the best backbone.

    The optimal backbone covers the most edges possible.
    """
    if len(indept_sets) > 4:
        indept_sets = indept_sets[:4]
    traces = []
    for x, y in itertools.combinations(indept_sets, 2):
        nbunch = x | y
        subgraph = G.subgraph(nbunch)
        n_edges = subgraph.number_of_edges()
        n_vertices = subgraph.number_of_nodes()
        index_x = indept_sets.index(x)
        index_y = indept_sets.index(y)
        traces.append((n_edges, n_vertices, index_x, index_y))

    return heapq.nlargest(2, traces)
