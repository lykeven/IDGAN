# -*- coding: utf-8 -*-
__author__ = 'keven'

import numpy as np
import rnn
import gan_rnn
import gan_rnn_gcn
import gan_rnn_gcn_feature
import utils



if __name__ == '__main__':
    args = utils.parse_args()
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
    gumbel = args.gumbel

    if args.gcn == 0:
        model = gan_rnn.GAN_RNN(g_input_step, g_input_size, g_hidden_size, g_output_step, g_batch_size, g_rate, g_epochs,
                            d_input_step, d_input_size, d_hidden_size, d_batch_size, d_rate, d_epochs, num_epochs,
                            print_interval, num_epochs_test, args.attention, args.wgan, args.w_clip, gumbel, data_file)
    else:
        if args.feature == 0:
            model = gan_rnn_gcn.GAN_RNN_GCN(g_input_step, g_input_size, g_hidden_size, g_output_step, g_batch_size, g_rate,
                                    g_epochs, d_input_step, d_input_size, d_hidden_size, d_batch_size, d_rate, d_epochs,
                                    num_epochs, print_interval,num_epochs_test, args.attention, args.wgan, args.w_clip,
                                    num_support, gumbel, graph_file, data_file)
        else:
            model = gan_rnn_gcn_feature.GAN_RNN_GCN_Feature(g_input_step, g_input_size, g_hidden_size, g_output_step, g_batch_size,
                                    g_rate, g_epochs, d_input_step, d_input_size, d_hidden_size, d_batch_size, d_rate, d_epochs,
                                    num_epochs, print_interval,num_epochs_test, args.attention, args.wgan, args.w_clip,
                                    num_support, gumbel, graph_file, data_file)

    if args.baseline == 1:
        model = rnn.RNN(g_input_step, g_input_size, g_hidden_size, g_output_step, g_batch_size, g_rate, num_epochs,
                        print_interval, num_epochs_test, args.attention, data_file)
    model.build_model()
    model.train()

