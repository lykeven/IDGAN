# -*- coding: utf-8 -*-
__author__ = 'keven'

import numpy as np
import argparse
import gan_rnn
import gan_rnn_gcn
import rnn


def parse_args():
	parser = argparse.ArgumentParser(description="Run ComEmbed.")

	parser.add_argument('-graph_file', nargs='?', default='graph.txt',
						help='Graph path')
	parser.add_argument('-data_file', nargs='?', default='diffusion.pkl',
						help='Input diffusion data')
	parser.add_argument('-g_input_step', type=int, default=15,
						help='Length of diffusion instance for generator. Default is 15.')
	parser.add_argument('-g_input_size', type=int, default=181,
						help='Number of nodes. Default is 181.')
	parser.add_argument('-g_hidden_size', type=int, default=64,
						help='Number of neurons at hidden layer. Default is 64.')
	parser.add_argument('-g_output_step', type=int, default=30,
						help='Length of diffusion instance for generator. Default is 128.')
	parser.add_argument('-g_batch_size', type=int, default=128,
						help='Size of a minibatch sample. Default is 128.')

	parser.add_argument('-d_input_step', type=int, default=30,
						help='Length of diffusion instance for discriminator. Default is 30.')
	parser.add_argument('-d_input_size', type=int, default=181,
						help='Number of nodes. Default is 181.')
	parser.add_argument('-d_hidden_size', type=int, default=64,
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

	return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    graph_file = args.graph_file
    data_file = args.data_file

    g_input_step = args.g_input_step
    g_input_size = args.g_input_size
    g_hidden_size = args.g_hidden_size
    g_output_step = args.g_output_step
    g_batch_size = args.g_batch_size

    d_input_step = args.d_input_step
    d_input_size = args.d_input_size
    d_hidden_size = args.d_hidden_size
    d_batch_size = args.d_batch_size

    g_rate = args.g_rate
    d_rate = args.d_rate
    g_epochs = args.g_epochs
    d_epochs = args.d_epochs

    num_epochs = args.num_epochs
    num_epochs_test = args.num_epochs_test
    print_interval = args.print_interval
    num_support = args.num_support

    if args.gcn == 0:
        model = gan_rnn.GAN_RNN(g_input_step, g_input_size, g_hidden_size, g_output_step, g_batch_size, g_rate, g_epochs,
                            d_input_step, d_input_size, d_hidden_size, d_batch_size, d_rate, d_epochs, num_epochs,
                            print_interval, num_epochs_test, data_file)
    else:
        model = gan_rnn_gcn.GAN_RNN_GCN(g_input_step, g_input_size, g_hidden_size, g_output_step, g_batch_size, g_rate,
                                    g_epochs, d_input_step, d_input_size, d_hidden_size, d_batch_size, d_rate, d_epochs,
                                    num_epochs, print_interval,num_epochs_test, num_support, graph_file, data_file)
    # model = rnn.RNN(g_input_step, g_input_size, g_hidden_size, g_output_step, g_batch_size, g_rate, num_epochs,
    #                 print_interval, num_epochs_test, data_file)
    model.build_model()
    model.train()

