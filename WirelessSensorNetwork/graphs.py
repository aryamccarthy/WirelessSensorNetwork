"""Random geometric graph.

We needed more functionality than networkx's graph generator.
"""

from copy import deepcopy

import numpy as np
import networkx as nx
from scipy.spatial import KDTree

import randomspatial as rs


def get_radius_from_degree(n, avg_deg, shape):
    if shape == "sphere":
        return np.sqrt(4 * avg_deg / n)
    elif shape == "disk":
        return np.sqrt(avg_deg / n)
    elif shape == "square":
        return np.sqrt(avg_deg / (n * np.pi))


def graph_with_distribution(n, avg_degree, shape):
    if shape == "disk":
        generator = rs.in_disk
    elif shape == "square":
        generator = rs.in_square
    elif shape == "sphere":
        generator = rs.on_sphere
    coordinates = {i: generator() for i in range(n)}
    radius = get_radius_from_degree(n, avg_degree, shape)
    return RandomGeometricGraph(n, radius, shape, coordinates)


class RandomGeometricGraph(nx.Graph):
    """Utilities for constructing a random geometric graph."""
    def __init__(self, n, radius, shape, pos, data=None, **attrs):
        super(RandomGeometricGraph, self).__init__(data, **attrs)
        self.name = "Random Geometric Graph"
        self._radius = radius
        self._shape = shape
        self.add_nodes_from(range(n))
        nx.set_node_attributes(self, 'pos', pos)
        self.add_edges_from(self._construct_edges())

    @property
    def pos(self):
        return nx.get_node_attributes(self, 'pos')

    def average_degree(self):
        return sum(deg for _, deg in self.degree()) / len(self)

    def estimated_average_degree(self):
        shape = self._shape
        n = len(self)
        r = self._radius
        if shape == 'disk':
            return n * r ** 2
        elif shape == 'sphere':
            return 0.25 * n * r ** 2
        elif shape == 'square':
            return n * np.pi * r ** 2
        else:
            return ValueError("What shape is this graph?")

    def _construct_edges(self):
        dist_list = [self.pos[i] for i in range(len(self.pos))]

        kdtree = KDTree(dist_list)
        edges = kdtree.query_pairs(self._radius)
        return edges

    def subgraph(self, nbunch, copy=True):
        # subgraph is needed here since it can destroy edges in the
        # graph (copy=False) and we want to keep track of all changes.
        #
        # Also for copy=True Graph() uses dictionary assignment for speed
        # Here we use H.add_edge()
        #
        # Adapted from the NetworkX examples:
        # https://github.com/networkx/networkx/blob/master/examples/subclass/printgraph.py
        bunch = set(self.nbunch_iter(nbunch))

        if not copy:
            # remove all nodes (and attached edges) not in nbunch
            self.remove_nodes_from([n for n in self if n not in bunch])
            self.name = "Subgraph of (%s)" % (self.name)
            return self
        else:
            # create new graph and copy subgraph into it
            H = nx.Graph()
            H.name = "Subgraph of (%s)" % (self.name)
            # add nodes
            H.add_nodes_from(bunch)
            # add edges
            seen = set()
            for u, nbrs in self.adjacency():
                if u in bunch:
                    for v, datadict in nbrs.items():
                        if v in bunch and v not in seen:
                            dd = deepcopy(datadict)
                            H.add_edge(u, v, *dd)
                    seen.add(u)
            # copy node and graph attr dicts
            H.node = dict((n, deepcopy(d))
                          for (n, d) in self.node.items() if n in H)
            H.graph = deepcopy(self.graph)
            return H
