from __future__ import division
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import itertools
from collections import defaultdict
from operator import itemgetter
import heapq
from scipy import spatial
import cProfile
import time
from mpl_toolkits.mplot3d import Axes3D


def do_cprofile(func):
    def profiled_func(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = func(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats()
    return profiled_func


np.random.seed(1337)


def distance(x0, x1):
    return np.linalg.norm(x1 - x0)


def magnitude(x):
    return np.linalg.norm(x)


def uniform_sample(a, b, dim):
    return (b - a) * np.random.random_sample(dim) + a


def uniform_sample_on_surface():
    u, v = np.random.random_sample(2)
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    return np.asarray([
        np.sin(phi) * np.cos(theta),  # x
        np.sin(phi) * np.sin(theta),  # y
        np.cos(phi)                   # z
    ])


def uniform_in_disk():
    a, b = np.random.random_sample(2)
    if b < a:
        a, b = b, a
    return np.array([b * np.cos(2 * np.pi * a / b),
                     b * np.sin(2 * np.pi * a / b)])


def construct_edgelist(distribution, r):
    dist_list = [distribution[i] for i in xrange(len(distribution))]
    kdtree = spatial.KDTree(dist_list)
    pairs = kdtree.query_pairs(r)
    return list(pairs)


def generate_coordinates_on_sphere(n):
    return {i: uniform_sample_on_surface() for i in xrange(n)}


def generate_coordinates_on_disk(n):
    return {i: uniform_in_disk() for i in xrange(n)}


def generate_coordinates_on_square(n):
    return {i: uniform_sample(0, 1, dim=2) for i in xrange(n)}


def get_r_from_deg(n, avg_deg, shape):
    if shape == "sphere":
        return np.sqrt(4 * avg_deg / n)
    elif shape == "disk":
        return np.sqrt(avg_deg / n)
    elif shape == "square":
        return np.sqrt(avg_deg / (n * np.pi))


def graph_from_inputs(n, avg_degree, shape):
    return graph_with_distribution(n,
                                   get_r_from_deg(n, avg_degree, shape),
                                   shape)


def rgg(n, radius, dim=2, pos=None):
    G = nx.Graph()
    G.name = "Random Geometric Graph"
    G.add_nodes_from(xrange(n))
    if pos is None:
        # Random positions
        pos = {i: np.random.random_sample(dim) for i in xrange(n)}
    nx.set_node_attributes(G, 'pos', pos)
    G.add_edges_from(construct_edgelist(pos, radius))
    return G


def graph_with_distribution(n, radius, shape):
    if shape == "disk":
        distribution = generate_coordinates_on_disk(n)
    elif shape == "square":
        distribution = generate_coordinates_on_square(n)
    elif shape == "sphere":
        distribution = generate_coordinates_on_sphere(n)
    G = rgg(n, radius, pos=distribution)
    # G = nx.random_geometric_graph(n, radius, pos=distribution)
    G.graph['shape'] = shape
    G.graph['radius'] = radius
    G.graph['n'] = n
    avg_deg = average_degree(G)

    print "|V|: {}\t|E|: {}\tR: {}\tAvg. Deg.: {}".format(G.number_of_nodes(),
                                                          G.number_of_edges(),
                                                          radius, avg_deg)
    return G


def average_degree(G):
    return sum((deg for (node, deg) in G.degree_iter())) / G.number_of_nodes()


def estimated_average_degree(G):
    shape = G.graph['shape']
    n = G.graph['n']
    r = G.graph['radius']
    if shape == "sphere":
        return 0.25 * n * r ** 2
    elif shape == "disk":
        return n * r ** 2
    elif shape == "square":
        return n * np.pi * r ** 2
    else:
        return ValueError("What shape is this graph?")


def overlay_graph(G):
    nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), node_size=10)
    plt.savefig("simplegraph{0}.pdf".format(ROUND))


def simple_show_graph(G):
    fig = plt.figure()
    try:
        nx.draw(G, pos=nx.get_node_attributes(G, 'pos'), node_size=10)
    except AssertionError:
        X, Y, Z = zip(*nx.get_node_attributes(G, 'pos').itervalues())
        ax = ax = Axes3D(fig)  # fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z)
    finally:
        plt.savefig("simplegraph{0}.pdf".format(ROUND))
    # plt.show()


def show_graph(G):
    min_degree = min((G.degree(x) for x in G.nodes_iter()))
    max_degree = max((G.degree(x) for x in G.nodes_iter()))
    nodes_with_min_degree = [x for x in G.nodes_iter()
                             if G.degree(x) == min_degree]
    nodes_with_max_degree = [x for x in G.nodes_iter()
                             if G.degree(x) == max_degree]
    node_color = ["k"] * G.number_of_nodes()
    for x in nodes_with_min_degree:
        node_color[x] = 'b'
    for x in nodes_with_max_degree:
        node_color[x] = 'r'

    min_edges = nx.edges(G, nodes_with_min_degree[0])
    max_edges = nx.edges(G, nodes_with_max_degree[0])

    fig = plt.figure()
    if average_degree(G) <= 30:
        nx.draw_networkx(G,
                         with_labels=True,
                         node_color=node_color,
                         width=0.1,
                         pos=nx.get_node_attributes(G, 'pos'),
                         node_size=10)
    else:
        nx.draw_networkx_nodes(G,
                               with_labels=False,
                               node_color=node_color,
                               pos=nx.get_node_attributes(G, 'pos'),
                               node_size=10)
    try:
        nx.draw_networkx_edges(G,
                               pos=nx.get_node_attributes(G, 'pos'),
                               edgelist=min_edges, edge_color='b')
        nx.draw_networkx_edges(G,
                               pos=nx.get_node_attributes(G, 'pos'),
                               edgelist=max_edges,
                               edge_color='r')
    except AssertionError:
        plt.clf()
        X, Y, Z = zip(*nx.get_node_attributes(G, 'pos').itervalues())
        ax = ax = Axes3D(fig)  # fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(X, Y, Z)  # ax.scatter(X, Y, Z)

    plt.savefig("graph{0}.pdf".format(ROUND))
    # plt.show()


def plot_degree_histogram(G):
    degree_sequence = nx.degree(G).values()
    dmax = np.int_(max(degree_sequence))
    dmin = np.int_(min(degree_sequence))
    plt.figure()
    plt.hist(degree_sequence, bins=dmax - dmin + 1)
    plt.title("Degree rank plot")
    plt.ylabel("degree")
    plt.xlabel("rank")
    plt.savefig("degrees{0}.pdf".format(ROUND))
    # plt.show()


def plot_sequential_coloring(degree_pairs):
    shrunken, original = itertools.izip(*degree_pairs)
    plt.figure()
    shr, = plt.plot(shrunken, 'b-', marker="", label="Smallest-last deg.")
    orig, = plt.plot(original, 'r-', marker=" ", label="Actual degree")
    plt.title("Smallest-last sizes")
    plt.ylabel("degree")
    plt.xlabel("rank")

    plt.legend([shr, orig])
    plt.savefig("ordering{0}.pdf".format(ROUND))
    # plt.show()


def min_degree_node(G):
    return min(G, key=G.degree)


# @do_cprofile
def smallest_last_ordering(G):
    len_g = len(G)
    H = G.copy(with_data=False)
    nodes = [None] * len_g
    degree_pairs = [None] * len_g

    available_degree = H.degree
    orig_degree = G.degree
    available_remove = H.remove_node

    # selection_list = H.nodes()
    degrees = defaultdict(set)
    for (node, degree) in H.degree_iter():
        degrees[degree].add(node)

    tcs = None

    for i in xrange(len_g):
        # min_deg = min(degrees)  # Gets smallest degree.
        for min_deg in itertools.count():
            if min_deg in degrees:
                break
        if i % 1000 == 0:
            print ".",
        # print degrees
        node = degrees[min_deg].pop()
        # print node
        if not degrees[min_deg]:
            del degrees[min_deg]
        for nbr in H.neighbors_iter(node):
            deg = available_degree(nbr)
            degrees[deg].remove(nbr)
            if not degrees[deg]:
                del degrees[deg]
            degrees[deg - 1].add(nbr)
        if len(degrees) == 1 and tcs is None:
            tcs = min(degrees)
        # selection_list.sort(key=available_degree, reverse=True)
        # node = selection_list.pop()# min(H, key=available_degree)
        original_degree = orig_degree(node)
        shrunken_degree = available_degree(node)

        available_remove(node)
        nodes[len_g - i - 1] = node

        degree_pairs[len_g - i - 1] = (shrunken_degree, original_degree)

    print ""
    return (nodes, degree_pairs, tcs)


def greedy_color(G, nodes=None):
    colors = {}

    if len(G):
        if nodes is None:
            nodes = smallest_last_ordering(G)[0]

        if nodes:
            for node in nodes:
                # set to track neighbors' colors
                neighbor_colors = set()
                for neighbor in G.neighbors_iter(node):
                    if neighbor in colors:
                        neighbor_colors.add(colors[neighbor])
                for color in itertools.count():
                    if color not in neighbor_colors:
                        break
                colors[node] = color

    return colors


def count_faces(G):
    # F + V = E + C + 1 where C is the number of components.
    # So F = E - V + C + 1
    E = nx.number_of_edges(G)
    V = nx.number_of_nodes(G)
    C = nx.number_connected_components(G)
    return E - V + C + 1


def select_backbones(G, indept_sets):
    if len(indept_sets) < 4:
        indept_sets = indept_sets[:4]

    traces = []
    for (x, y) in itertools.combinations(indept_sets, 2):
        nbunch = x | y
        subgraph = G.subgraph(nbunch)
        n_edges = subgraph.number_of_edges()
        n_vertices = subgraph.number_of_nodes()
        index1 = indept_sets.index(x)
        index2 = indept_sets.index(y)
        traces.append((n_edges, n_vertices, index1, index2))

    return heapq.nlargest(2, traces)


def get_nodes_for_colors(colors):
    lst = [set() for _ in xrange(1 + max(colors.itervalues()))]
    for (node, color) in colors.iteritems():
        lst[color].add(node)
    return lst


def plot_color_counts(color_counts):
    plt.figure()
    idx, ct = itertools.izip(*enumerate(color_counts))
    plt.bar(idx, ct)
    plt.title("Color class size distribution")
    plt.xlabel("Color class")
    plt.ylabel("Size")
    plt.savefig("colors{0}.pdf".format(ROUND))
    # plt.show()


def draw_backbone(G, subgraph):
    if G.graph['n'] <= 4000:
        show_graph(G)
        overlay_graph(subgraph)
    else:
        simple_show_graph(subgraph)
    global ROUND
    ROUND = (ROUND * 10) + 1


def walkthrough():
    n = 20
    r = 0.4
    a = graph_with_distribution(n, r, "square")
    show_graph(a)


data = [
    {"id": 1, "n": 1000, "avg_deg": 30, "shape": "square"},
    {"id": 2, "n": 4000, "avg_deg": 40, "shape": "square"},
    {"id": 3, "n": 4000, "avg_deg": 60, "shape": "square"},
    {"id": 4, "n": 16000, "avg_deg": 60, "shape": "square"},
    {"id": 5, "n": 64000, "avg_deg": 60, "shape": "square"},
    {"id": 6, "n": 4000, "avg_deg": 60, "shape": "disk"},
    {"id": 7, "n": 4000, "avg_deg": 120, "shape": "disk"},
    {"id": 8, "n": 4000, "avg_deg": 60, "shape": "sphere"},
    {"id": 9, "n": 16000, "avg_deg": 120, "shape": "sphere"},
    {"id": 10, "n": 64000, "avg_deg": 120, "shape": "sphere"},
    {"id": 12, "n": 256000, "avg_deg": 120, "shape": "square"},
    {"id": 40, "n": 100000, "avg_deg": 60, "shape": "square"},
    {"id": 100, "n": 20, "avg_deg": 20 * np.pi * 0.4 ** 2, "shape": "square"},
    {"id": 50, "n": 1000, "avg_deg": 200, "shape": "square"},
]

ROUND = -1


def main():
    a = None
    for bench in data[-1:]:
        print "\n\n\nTRIAL {0}".format(bench)
        global ROUND
        ROUND = bench['id']
        # Part A
        print "Part A"
        t = time.time()
        a = graph_from_inputs(bench['n'], bench['avg_deg'], bench['shape'])
        t1 = time.time()
        print "Time: {0} m {1} s".format(int((t1 - t) // 60),
                                         int((t1 - t) % 60))
        show_graph(a)
        plot_degree_histogram(a)
        # print a.nodes(data=True)

        # Part B
        print "Part B"
        print "Smallest Last..."
        t = time.time()
        nodes, degree_pairs, tcs = smallest_last_ordering(a)
        # print nodes
        t1 = time.time()
        print "Time: {} m {} s".format(int((t1 - t) // 60),
                                       int((t1 - t) % 60))
        plot_sequential_coloring(degree_pairs)
        print "Greedy color..."
        colors = greedy_color(a, nodes)
        print "Indep't sets..."
        t = time.time()
        indept_sets = get_nodes_for_colors(colors)
        print indept_sets
        color_counts = [len(s) for s in indept_sets]
        t1 = time.time()
        print "Time: {} m {} s".format(int((t1 - t) // 60),
                                       int((t1 - t) % 60))
        plot_color_counts(color_counts)
        print "max_min_degree..."
        max_min_degree = max(degree_pairs, key=itemgetter(0))[0]
        print ("ID: {0}\tN: {1}\tR: {2}\tShape: "
               "{3}\tM: {4}\tmin_deg: {5}\tavg_deg: "
               "{6}\tmax_deg: {7}\tmax_min: {8}\tcolors: {9}\tmax_c: "
               "{10}\tt_c_size: {11}".format(bench['id'],
                                             bench['n'],
                                             a.graph['radius'],
                                             bench['shape'],
                                             a.number_of_edges(),
                                             min(a.degree().itervalues()),
                                             average_degree(a),
                                             max(a.degree().itervalues()),
                                             max_min_degree,
                                             len(color_counts),
                                             max(color_counts),
                                             tcs))
        # Part C
        print "Part C"
        t = time.time()
        bbs = select_backbones(a, indept_sets)
        for bb in bbs:
            print "M: {}\tN: {}\t% {}".format(bb[0],
                                              bb[1],
                                              bb[1] / a.number_of_nodes())
            subgraph = a.subgraph(indept_sets[bb[2]] | indept_sets[bb[3]])
            if bench['shape'] != "sphere":
                draw_backbone(a, subgraph)
            else:  # It's a sphere!
                print "Faces: {0}".format(count_faces(subgraph))
                draw_backbone(a, subgraph)
        t1 = time.time()
        print "Time: {} m {} s".format(int((t1 - t) // 60),
                                       int((t1 - t) % 60))
        plt.close('all')


if __name__ == '__main__':
    main()
    # walkthrough()
