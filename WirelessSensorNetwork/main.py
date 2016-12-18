from collections import namedtuple
from operator import itemgetter

import numpy as np

import backbone
from graphs import graph_with_distribution
from visualize import (draw_graph, plot_degree_histogram,
                       plot_sequential_coloring, plot_color_counts, 
                       draw_backbone)
from utils import time_execution, Timer


Benchmark = namedtuple('Benchmark', ['n', 'avg_deg', 'shape'])
benchmarks = [
    # Required benchmarks
    Benchmark(1000, 30, 'square'),
    Benchmark(4000, 40, 'square'),
    Benchmark(4000, 60, 'square'),
    Benchmark(16000, 60, 'square'),
    Benchmark(64000, 60, 'square'),
    Benchmark(4000, 60, 'disk'),
    Benchmark(4000, 120, 'disk'),
    Benchmark(4000, 60, 'sphere'),
    Benchmark(16000, 120, 'sphere'),
    Benchmark(64000, 120, 'sphere'),
    # Custom benchmarks
    Benchmark(256000, 120, 'square'),
    Benchmark(100000, 60, 'square'),
    Benchmark(20, 20 * np.pi * 0.4 ** 2, 'square'),
    Benchmark(1000, 200, 'square'),
]


@time_execution
def graph_for_benchmark(benchmark):
    return graph_with_distribution(benchmark.n, benchmark.avg_deg,
                                   benchmark.shape)


def main(verbose=True):
    np.random.seed(1337)

    for idx, benchmark in enumerate(benchmarks[:1]):
        print("\n\n\nTrial {}: {}".format(idx, benchmark))

        print("Part A")
        G = graph_for_benchmark(benchmark)
        draw_graph(G)
        plot_degree_histogram(G)

        print("Part B")
        (nodes, degree_pairs,
            tcs) = backbone.smallest_last_ordering(G, with_stats=True)
        plot_sequential_coloring(degree_pairs)

        colors = backbone.greedy_color(G, nodes)

        with Timer():
            indept_sets = backbone.independent_sets_from_colors(colors)
            color_counts = [len(s) for s in indept_sets]
        plot_color_counts(color_counts)

        max_min_degree = max(degree_pairs, key=itemgetter(0))[0]
        print(("ID: {0}\tN: {1}\tR: {2}\tShape: "
               "{3}\tM: {4}\tmin_deg: {5}\tavg_deg: "
               "{6}\tmax_deg: {7}\tmax_min: {8}\tcolors: {9}\tmax_c: "
               "{10}\tt_c_size: {11}".format(idx,
                                             benchmark.n,
                                             G._radius,
                                             G._shape,
                                             G.number_of_edges(),
                                             min(d for i, d in G.degree()),
                                             G.average_degree(),
                                             max(d for i, d in G.degree()),
                                             max_min_degree,
                                             len(color_counts),
                                             max(color_counts),
                                             tcs)))

        print("Part C")
        with Timer():
            subgraphs = []
            bbs = backbone.select_backbones(G, indept_sets)
            for bb in bbs:
                print("M: {}\tN: {}\t% {}".format(bb[0], bb[1],
                                                  bb[1] / len(G)))
                print("Subgraphs...")
                subgraph = G.subgraph(indept_sets[bb[2]] |
                                      indept_sets[bb[3]])
                subgraphs.append(subgraph)
                print("Drawing...")
                if benchmark.shape == 'sphere':
                    print("Faces: {0}".format(backbone.count_faces(subgraph)))
        for bb in subgraphs:
            draw_backbone(G, bb)


if __name__ == '__main__':
    main()
