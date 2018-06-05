#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 10:42:52 2018

@author: kenn
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)


# 思路：把一行像素作为一个x，把每一行作为一个输入节点，用RNN分类图片
# hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

n_inputs = 28 # 输入的一行的长度（每个输入节点的维度）
n_steps = 28 # RNN的长度（输入节点的个数）
n_hidden_units = 128
n_classes = 10


# 输入层
x = tf.placeholder(tf.float32, [None,n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None,n_classes])

cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden_units)

state = cell.zero_state(batch_size, tf.float32)
state = tf.contrib.rnn.LSTMStateTuple(
            tf.Variable(state[0], trainable=False),
            tf.Variable(state[1], trainable=False))

outputs, new_state = tf.nn.dynamic_rnn(cell,inputs= x,
                                       initial_state= state, 
                                       dtype=tf.float32, 
                                       time_major=False
                                       )

with tf.control_dependencies([state[0].assign(new_state[0]), state[1].assign(new_state[1])]):
    outputs = tf.identity(outputs)

output = tf.layers.dense(outputs[:,-1,:],10,tf.nn.softmax)

cost = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=output)
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        #x_image = tf.reshape(xs,[-1,28,28]) 也可以在placeholder下定义reshape
        sess.run(train_op, feed_dict = {x:batch_xs, y:batch_ys})
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict = {x:batch_xs, y:batch_ys}))
        step+=1
