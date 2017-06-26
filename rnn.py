# -*- coding: utf-8 -*-
__author__ = 'keven'

import numpy as np
import tensorflow as tf
import sys
import utils

# weight initialize
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


class RNN():
    def __init__(self, input_step=14, input_size=28, hidden_size=50, output_step=28, batch_size=50, rate=2e-4,
                 epochs=1,  print_interval=10, num_epochs_test=30, attention=0, data_file="diffusion.pkl"):
        self.input_step = input_step
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_step = output_step
        self.batch_size = batch_size
        self.rate = rate
        self.epochs = epochs

        self.print_interval = print_interval
        self.num_epochs_test = num_epochs_test
        self.attention = attention
        self.data_file = data_file


    def generator(self, input, input_step, input_size, hidden_size, batch_size, reuse=False):
        with tf.variable_scope("generator") as scope:
            # lstm cell and wrap with dropout
            g_lstm_cell = tf.contrib.rnn.BasicLSTMCell(input_size, forget_bias=0.0, state_is_tuple=True)
            g_lstm_cell_1 = tf.contrib.rnn.BasicLSTMCell(input_size, forget_bias=0.0, state_is_tuple=True)

            g_lstm_cell_attention = tf.contrib.rnn.AttentionCellWrapper(g_lstm_cell, attn_length=10)
            g_lstm_cell_attention_1 = tf.contrib.rnn.AttentionCellWrapper(g_lstm_cell_1, attn_length=10)

            if self.attention == 1:
                g_lstm_cell_drop = tf.contrib.rnn.DropoutWrapper(g_lstm_cell_attention, output_keep_prob=0.9)
                g_lstm_cell_drop_1 = tf.contrib.rnn.DropoutWrapper(g_lstm_cell_attention_1, output_keep_prob=0.9)
            else:
                g_lstm_cell_drop = tf.contrib.rnn.DropoutWrapper(g_lstm_cell, output_keep_prob=0.9)
                g_lstm_cell_drop_1 = tf.contrib.rnn.DropoutWrapper(g_lstm_cell_1, output_keep_prob=0.9)

            g_cell = tf.contrib.rnn.MultiRNNCell([g_lstm_cell_drop, g_lstm_cell_drop_1], state_is_tuple=True)
            g_state_ = g_cell.zero_state(batch_size, tf.float32)
            # g_W_o = utils.glorot([hidden_size, input_size])
            # g_b_o = tf.Variable(tf.random_normal([input_size]))

            # neural network
            g_outputs = []
            g_state = g_state_
            for i in range(input_step):
                if i > 0: tf.get_variable_scope().reuse_variables()
                (g_cell_output, g_state) = g_cell(input[:, i, :], g_state)  # cell_out: [batch_size, hidden_size]
                g_outputs.append(g_cell_output)  # output: shape[input_step][batch_size, hidden_size]

            # expend outputs to [batch_size, hidden_size * input_step] and then reshape to [batch_size * input_steps, hidden_size]
            g_output = tf.reshape(tf.concat(g_outputs, axis=1), [-1, input_size])
            g_y_soft = tf.nn.softmax(g_output)
            g_ = tf.reshape(g_y_soft, [batch_size, input_step, input_size])
            return g_


    def build_model(self,):
        self.z = tf.placeholder(tf.float32, [None, self.input_step, self.input_size])
        self.z_t = tf.placeholder(tf.float32, [None, self.input_step, self.input_size])
        self.z_ = self.generator(self.z, self.input_step, self.input_size, self.hidden_size, self.batch_size)

        # self.loss = tf.losses.mean_squared_error(self.z_, self.z_t)
        labels = tf.reshape(self.z_t, [self.batch_size * self.input_step, self.input_size])
        logits = tf.reshape(self.z_, [self.batch_size * self.input_step, self.input_size])
        self.loss = tf.losses.softmax_cross_entropy(onehot_labels=labels, logits=logits)


        def compute_accuracy(x, y):
            intersection = tf.sets.set_intersection(tf.argmax(x, 2), tf.argmax(y, 2))
            correct_number = tf.reduce_sum(tf.sets.set_size(intersection))
            return tf.cast(correct_number, tf.float32) / self.input_step / self.batch_size

        self.accuracy = compute_accuracy(self.z_t, self.z_)


    def train(self,):
        utils.prepare_data(data_file=self.data_file)
        optim = tf.train.RMSPropOptimizer(self.rate).minimize(self.loss)

        init = tf.global_variables_initializer()
        sess = tf.InteractiveSession()
        sess.run(init)

        for i in range(self.epochs):
            batch_z, batch_x, batch_z_ = utils.feed_data(self.batch_size, self.input_step, self.input_size)
            z_ = sess.run(self.z_,feed_dict={self.z: batch_z})
            sess.run(optim, feed_dict={self.z: batch_z, self.z_t: batch_z_})
            if i % self.print_interval == 0:
                cost = sess.run(self.loss, feed_dict={self.z: batch_z, self.z_t: batch_z_})
                accuracy = sess.run(self.accuracy, feed_dict={self.z: batch_z, self.z_t: batch_z_})
                print "Iter %d, loss = %.5f, accuracy = %.5f" % (i, cost, accuracy)

        # test performance
        loss_list = []
        accuracy_list =[]
        for i in range(self.num_epochs_test):
            test_z, test_x, test_z_ = utils.feed_data(self.batch_size, self.input_step, self.input_size, is_train=False)
            cost = sess.run(self.loss, feed_dict={self.z: test_z, self.z_t:test_z_})
            accuracy = sess.run(self.accuracy, feed_dict={self.z: test_z, self.z_t:test_z_})
            loss_list.append(cost)
            accuracy_list.append(accuracy)
        print "Testing Loss: loss = %.5f, accuracy = %.5f" % (sum(loss_list)/len(loss_list), sum(accuracy_list)/ len(accuracy_list))



