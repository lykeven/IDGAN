# -*- coding: utf-8 -*-
__author__ = 'keven'

import sys
import numpy as np
import networkx as nx
import pickle as cp
import scipy.sparse as sp
import tensorflow as tf
from sklearn.preprocessing import normalize
from scipy.sparse.linalg.eigen.arpack import eigsh
import argparse


def read_network(max_node_num=10000):
    # read the network from input file
    G = nx.DiGraph()
    edges = []
    network_file = "weibo_network.txt"
    f = open(network_file)
    num_nodes, num_edges = map(int, f.readline().strip().split("\t"))
    for i in range(max_node_num):
        line = map(int, f.readline().strip().split("\t"))
        v_id, v_num_edges = line[0], line[1]
        new_edges = [(v_id, line[2 + 2 * i]) for i in range(v_num_edges) if line[2 + 2 * i]<max_node_num]
        di_edges = [(line[2 + 2 * i], v_id) for i in range(v_num_edges) if line[2 + 2 * i]<max_node_num and line[3 + 2 * i]==1]
        edges += new_edges
        edges += di_edges
    f.close()
    G.add_edges_from(edges)
    print("network read done")
    return G


def read_map(max_node_num=10000):
    node2id = dict()
    map_file = "uidlist.txt"
    f = open(map_file)
    for i, line in enumerate(f):
        if i >= max_node_num:break
        node2id[line.strip()] = i
    f.close()
    print("map read done")
    return node2id


def read_feature(node2id, max_node_num=10000):
    info_dict = dict()
    feature_file1 = "user_profile1.txt"
    feature_file2 = "user_profile2.txt"
    num_lines = 15
    i = 0
    lines = open(feature_file1).readlines()
    while i <len(lines):
        if i == 0: i += num_lines
        ct = lines[i:i + num_lines]
        if ct[0].strip() in node2id:
            info = [int(ct[1].strip()), int(ct[4].strip()), int(ct[7].strip()), int(ct[12].strip())]
            info_dict[node2id[ct[0].strip()]] = info
        i += num_lines

    i = 0
    lines = open(feature_file2).readlines()
    while i <len(lines):
        ct = lines[i:i + num_lines]
        if ct[0].strip() in node2id:
            info = [int(ct[1].strip()), int(ct[4].strip()), int(ct[7].strip()), int(ct[12].strip())]
            info_dict[node2id[ct[0].strip()]] = info
        i += num_lines
    print("features read done")
    return info_dict


def read_diffusion(node2id, max_node_num=10000):
    m_info_dict = dict()
    m_retweet_dict = dict()
    diffusion_file = "total.txt"
    f = open(diffusion_file)
    iter = 0
    while 1:
        line = f.readline()
        if not line:
            break
        m_id, m_time, u_id, retweet_num = line.strip().split(" ")
        id_time_temp = f.readline().strip().split(" ")
        id_time_list = [[id_time_temp[2 * i], id_time_temp[2 * i + 1]] for i in range(len(id_time_temp) / 2)
                        if id_time_temp[2 * i] in node2id and node2id[id_time_temp[2 * i]]<max_node_num]
        if len(id_time_list) > 0:
            m_info_dict[m_id] = [m_time, u_id, int(retweet_num)]
            m_retweet_dict[m_id] = id_time_list
        iter += 1
        if iter % 10000 ==0:
            print ("%dth diffusion" % (iter,))
    f.close()
    print("diffusion read done")
    return m_info_dict, m_retweet_dict


def extract_sub_graph(graph, node2id, m_info, m_retweet, num_node=100, length=30):
    # extract a dense sub graph
    sub_retweet = []
    user_retweet = [[]] * len(node2id)
    for m_id, info in m_info.items():
        if info[2] > length:
            for user, time in m_retweet[m_id]:
                user_retweet[node2id[user]].append([m_id, time])
    print("user retweet collect done")

    degree_dict = graph.in_degree()
    ordered_degree = sorted(degree_dict.items(), key=lambda a: a[1], reverse=True)
    selected_node_list = [node[0] for node in ordered_degree[:num_node]]
    selected_node_dict = dict(zip(selected_node_list, [False] * len(selected_node_list)))
    for node in selected_node_list:
        for v, u in graph.out_edges(node):
            selected_node_dict[v] = True
            selected_node_dict[u] = True

    retweet_dict = dict()
    sub_graph = graph.subgraph([key for key, value in selected_node_dict.items() if value==True])
    print("subgraph extract done")

    for node in sub_graph.nodes():
        for m_id, time in user_retweet[node]:
            if m_id not in retweet_dict:
                retweet_dict[m_id] = [[node, time]]
            else:
                retweet_dict[m_id].append([node, time])

    print("message select done")
    for m_id, retweet_info in retweet_dict.items():
        while len(retweet_info) >= length:
            diff_node_list = sorted(retweet_info[:length], key= lambda a: a[1], reverse=False)
            sub_retweet.append([a[0] for a in diff_node_list])
            retweet_info = retweet_info[length:]
    return sub_graph, sub_retweet


def save_data(subgraph, sub_retweet, features, length=30, graph_file="graph.txt", diffusion_data_file="diffusion.pkl"):
    # save data to file
    node2id = dict([(node, vid) for vid, node in enumerate(subgraph.nodes())])
    print("remap node done")

    f_graph = open(graph_file, 'w')
    f_graph.write("\n".join([str(node2id[v]) + " " + str(node2id[u]) for v, u in subgraph.edges()]))
    f_graph.close()
    print("graph write done with %d nodes and %d edges" % (subgraph.number_of_nodes(), subgraph.number_of_edges()))

    diff_data = np.zeros((len(sub_retweet), length), dtype=np.int)
    for i, retweet in enumerate(sub_retweet):
        for j, node in enumerate(retweet):
            diff_data[i, j] = node2id[node]

    feature_matrix = np.zeros((len(node2id), len(features.values()[0])), dtype=np.float)
    mean = np.asarray(features.values()).mean(axis=0)
    for id, node in enumerate(subgraph.nodes()):
        if node in features:
            feature_matrix[id] = features[node]
        else:
            feature_matrix[id] = mean
    feature_matrix = normalize(feature_matrix, norm="l2", axis=0)
    print("feature write done with %d dimension" % (feature_matrix.shape[1]))

    diff_train = diff_data[:int(len(sub_retweet) * 0.8)]
    diff_test = diff_data[int(len(sub_retweet) * 0.8):]
    data = {"train":diff_train, "test":diff_test, "feature":feature_matrix}
    with open(diffusion_data_file, 'w') as f:
        cp.dump(data, f)
    print("diffusion write done with %d diffusion path" % (len(sub_retweet)))

    prepare_data()
    batch_data = train_next_batch(batch_size=128, input_size=subgraph.number_of_nodes())



train_pos = 0
test_pos = 0
all_data = None

def prepare_data(data_file="diffusion.pkl"):
    global all_data
    with open(data_file, 'r') as f:
        all_data = cp.load(f)


def get_diffusion_matrix(diff_batch, num_node=181):
    diff_data = np.zeros((diff_batch.shape[0], diff_batch.shape[1], num_node), dtype=np.float)
    for i in range(diff_data.shape[0]):
        for j in range(diff_data.shape[1]):
            diff_data[i, j, diff_batch[i, j]] = 1.0
    return diff_data


def train_next_batch(batch_size, input_size=181):
    global train_pos, all_data
    if train_pos + batch_size > all_data["train"].shape[0]:
        train_pos = 0
    batch_data = get_diffusion_matrix(all_data["train"][train_pos:train_pos + batch_size], input_size)
    train_pos += batch_size
    return batch_data


def test_next_batch(batch_size, input_size=181):
    global test_pos, all_data
    if test_pos + batch_size > all_data["test"].shape[0]:
        test_pos = 0
    batch_data = get_diffusion_matrix(all_data["test"][test_pos:test_pos + batch_size], input_size)
    test_pos += batch_size
    return batch_data


def load_gcn_data(filename, num_support):
    global all_data
    graph = nx.read_edgelist(filename, nodetype=int, create_using=nx.DiGraph())
    adj = nx.adjacency_matrix(graph)
    lap_list = chebyshev_polynomials(adj, k=num_support)
    features = all_data["feature"]
    return lap_list, features


def feed_data(batch_size, input_step, input_size, is_train=True):
    if is_train:
        batch_temp = train_next_batch(batch_size * 2, input_size)
    else:
        batch_temp = test_next_batch(batch_size * 2, input_size)
    batch_z = batch_temp[:batch_size, :input_step]
    batch_z_target = batch_temp[:batch_size, input_step:input_step * 2]
    batch_x = batch_temp[batch_size:batch_size * 2, :]
    return batch_z, batch_x, batch_z_target


def glorot(shape, name=None):
    # weight init
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return tf.Variable(initial, name=name)


def normalize_adj(adj):
    # Symmetrically normalize adjacency matrix
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def chebyshev_polynomials(adj, k):
    # Calculate Chebyshev polynomials up to order k.
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = np.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = np.zeros((k, adj.shape[0], adj.shape[1]), dtype=np.float)
    t_k[0] = np.eye(adj.shape[0])
    t_k[1] = np.asarray(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k):
        t_k[i] = chebyshev_recurrence(t_k[i-1], t_k[i-2], scaled_laplacian)
    return np.asarray(t_k)



def parse_args():
	parser = argparse.ArgumentParser(description="Run ComEmbed.")

	parser.add_argument('-graph_file', nargs='?', default='graph.txt',
						help='Graph path')
	parser.add_argument('-data_file', nargs='?', default='diffusion.pkl',
						help='Input diffusion data')
	parser.add_argument('-g_input_step', type=int, default=15,
						help='Length of diffusion instance for generator. Default is 15.')
	parser.add_argument('-g_input_size', type=int, default=725,
						help='Number of nodes. Default is 725.')
	parser.add_argument('-g_hidden_size', type=int, default=64,
						help='Number of neurons at hidden layer. Default is 64.')
	parser.add_argument('-g_output_step', type=int, default=30,
						help='Length of diffusion instance for generator. Default is 128.')
	parser.add_argument('-g_batch_size', type=int, default=128,
						help='Size of a minibatch sample. Default is 128.')

	parser.add_argument('-d_input_step', type=int, default=30,
						help='Length of diffusion instance for discriminator. Default is 30.')
	parser.add_argument('-d_input_size', type=int, default=725,
						help='Number of nodes. Default is 181.')
	parser.add_argument('-d_hidden_size', type=int, default=10,
						help='Number of neurons at hidden layer. Default is 64.')
	parser.add_argument('-d_batch_size', type=int, default=128,
						help='Size of a minibatch sample. Default is 128.')

	parser.add_argument('-g_rate', type=float, default=2e-2,
						help='Learning rate of SGD for generator. Default is 2e-2.')
	parser.add_argument('-d_rate', type=float, default=2e-2,
						help='Learning rate of SGD for discriminator. Default is 2e-2.')
	parser.add_argument('-g_epochs', type=int, default=1,
						help='Number of iteration for generator. Default is 1.')
	parser.add_argument('-d_epochs', type=int, default=2,
						help='Number of iteration for discriminator. Default is 2.')

	parser.add_argument('-num_epochs', type=int, default=3000,
						help='Number of iteration for gan. Default is 3000.')
	parser.add_argument('-num_epochs_test', type=int, default=30,
						help='Number of iteration for gan. Default is 30.')
	parser.add_argument('-print_interval', type=int, default=100,
						help='Interval of print information. Default is 100.')
	parser.add_argument('-num_support', type=int, default=5,
						help='Number of highest order laplacian matrix. Default is 5.')
	parser.add_argument('-gcn', type=int, default=0,
						help='Whether use GCN to train GAN model. Default is 0.')
	parser.add_argument('-feature', type=int, default=0,
						help='Whether use node attribute features in GCN to train GAN model. Default is 0.')
	parser.add_argument('-wgan', type=int, default=0,
						help='Whether use WGAN model. Default is 0.')
	parser.add_argument('-w_clip', type=float, default=1e-2,
						help='Clip parameter in WGAN model. Default is 1e-2.')
	parser.add_argument('-attention', type=int, default=0,
						help='Whether use attention to train GAN model. Default is 0.')
	parser.add_argument('-baseline', type=int, default=0,
						help='Whether train baseline model. Default is 0.')

	return parser.parse_args()


def main():
    max_node_num = 10000
    args = parse_args()
    graph = read_network(max_node_num)
    node2id = read_map(max_node_num)
    u_info = read_feature(node2id, max_node_num)
    m_info, m_retweet = read_diffusion(node2id, max_node_num)
    sub_graph, sub_retweet = extract_sub_graph(graph, node2id, m_info, m_retweet)
    save_data(sub_graph, sub_retweet, u_info, args.graph_file, args.data_file)

if __name__ == '__main__':
    sys.exit(main())


