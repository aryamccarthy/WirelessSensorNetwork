"""Show all kinds of cool plots."""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import numpy as np
import seaborn as sns


from utils import show_figure


def _draw_nodes(G):
    nx.draw_networkx_nodes(G, with_labels=False, 
                           pos=nx.get_node_attributes(G, 'pos'),
                           node_size=5)


def _draw_graph(G):
    plt.axis('equal')
    # Identify vertices with min and max degree.
    degrees = np.array([G.degree(v) for v in G])
    min_deg_nodes, = np.where(degrees == degrees.min())
    max_deg_nodes, = np.where(degrees == degrees.max())

    # Color the min and max degree nodes.
    node_color = ['k'] * len(G)
    for x in min_deg_nodes:
        node_color[x] = 'b'
    for x in max_deg_nodes:
        node_color[x] = 'r'

    # Identify edges connected to min and max nodes.
    min_edges = list(G.edges(min_deg_nodes))
    max_edges = list(G.edges(max_deg_nodes))

    if G.average_degree() <= 20:
        # Draw nodes and edges.
        nx.draw_networkx(G, with_labels=True, node_color=node_color,
                         width=0.1, pos=nx.get_node_attributes(G, 'pos'),
                         node_size=10)
    else:
        # Draw nodes only.
        nx.draw_networkx_nodes(G, with_labels=False, node_color=node_color,
                               pos=nx.get_node_attributes(G, 'pos'),
                               node_size=10)
    try:
        # Draw max and min node edges in color.
        nx.draw_networkx_edges(G, pos=nx.get_node_attributes(G, 'pos'),
                               edgelist=min_edges, edge_color='b')
        nx.draw_networkx_edges(G, pos=nx.get_node_attributes(G, 'pos'),
                               edgelist=max_edges, edge_color='r')
    except ValueError:
        # Redraw as 3D.
        plt.clf()
        X, Y, Z = zip(*nx.get_node_attributes(G, 'pos').values())
        ax = Axes3D()
        ax.plot_wireframe(X, Y, Z)


draw_graph = show_figure(_draw_graph)


@show_figure
def plot_degree_histogram(G):
    degree_sequence = [d for _, d in G.degree()]
    sns.distplot(degree_sequence)
    plt.title("Degree rank plot")
    plt.ylabel("degree")
    plt.xlabel("rank")


@show_figure
def plot_sequential_coloring(degree_pairs):
    shrunken, original = zip(*degree_pairs)
    shr, = plt.plot(shrunken, 'b-', marker="", label="Smallest-last deg.")
    orig, = plt.plot(original, 'r-', marker="", label="Actual degree")
    plt.title("Smallest-last sizes")
    plt.ylabel("degree")
    plt.xlabel("rank")

    plt.legend([shr, orig])


@show_figure
def plot_color_counts(counts):
    idx, ct = zip(*enumerate(counts))
    plt.bar(idx, ct)
    plt.title("Color class size distribution")
    plt.xlabel("Color class")
    plt.ylabel("Size")


def _overlay_graph(G):
    nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), node_size=10)


def _simple_draw_graph(G):
    try:
        nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), node_size=10)
    except ValueError:
        X, Y, Z = list(zip(*(nx.get_node_attributes(G, 'pos').values())))
        ax = Axes3D()
        ax.scatter(X, Y, Z)


@show_figure
def draw_backbone(G, backbone):
    plt.axis('equal')
    if len(G) <= 4000:
        _draw_nodes(G)
        _overlay_graph(backbone)
    else:
        _simple_draw_graph(backbone)
