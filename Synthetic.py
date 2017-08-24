# -*- coding: utf-8 -*-
__author__ = 'keven'

import numpy as np
import networkx as nx
import pickle as cp
import sys
import utils


def sample_node(neighbor_dict):
    node_list, fre_list = neighbor_dict.keys(), neighbor_dict.values()
    sample_number = np.random.randint(0, len(fre_list))
    index = 0
    cur_number = 0
    for i, fre in enumerate(fre_list):
        cur_number += fre
        if cur_number >= sample_number:
            index = i
            break
    return node_list[index]


def generate_samples(graph, num_sample, length):
    samples = []
    num_node = graph.number_of_nodes()
    for i in range(num_sample):
        start_node = np.random.randint(0, num_node)
        start_neighbor = dict()
        diffusion_set = [start_node]
        for u, v in graph.in_edges(start_node):
            start_neighbor[u] = 1

        for j in range(length - 1):
            new_node = sample_node(start_neighbor)
            diffusion_set.append(new_node)
            del start_neighbor[new_node]

            for u, v in graph.in_edges(new_node):
                if u not in diffusion_set:
                    if u not in start_neighbor:
                        start_neighbor[u] = 1
                    else:
                        start_neighbor[u] += 1
        samples.append(diffusion_set)
    return samples



def save_data(graph, train_samples, test_samples, graph_file, data_file,):

    f_graph = open(graph_file, 'w')
    f_graph.write("\n".join([str(v) + " " + str(u) for v, u in graph.edges()]))
    f_graph.close()
    print("graph write done with %d nodes and %d edges" % (graph.number_of_nodes(), graph.number_of_edges()))


    train_data = np.zeros((len(train_samples), len(train_samples[0])), dtype=np.int)
    for i, sample in enumerate(train_samples):
        for j, node in enumerate(sample):
            train_data[i, j] = node

    test_data = np.zeros((len(test_samples), len(test_samples[0])), dtype=np.int)
    for i, sample in enumerate(test_samples):
        for j, node in enumerate(sample):
            test_data[i, j] = node
    data = {"train":train_data, "test":test_data}
    with open(data_file, 'w') as f:
        cp.dump(data, f)
    print("diffusion samples write done with %d train samples and %d test samples" % (len(train_samples), len(test_samples)))


def main():
    args = utils.parse_args_new()
    data_file = args.data_file
    graph_file = args.graph_file
    num_train_sample = args.num_train_sample
    num_test_sample = args.num_test_sample
    seq_length = args.seq_length
    num_node = args.num_node
    prob_edge = 0.02
    num_edge = int(num_node * (num_node - 1) * prob_edge)
    graph = nx.random_graphs.gnm_random_graph(num_node, num_edge, directed=True)
    train_samples = generate_samples(graph, num_train_sample, seq_length)
    test_samples = generate_samples(graph, num_test_sample, seq_length)
    save_data(graph, train_samples, test_samples, graph_file, data_file)

if __name__ == '__main__':
    sys.exit(main())