#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 10:42:52 2018

@author: kenn
"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data",one_hot=True)


drop_prob = 0.5
filters_1 = 6
filters_2 = 16
filters_3 = 120
fc_units = 84
kernel_size = 5
pool_size = 2
conv_strides = 1
pool_strides = 2

inputs_size = 28
input_channel = 1
output_size = 10


def compute_accuracy(v_xs, v_ys):
    global prediction #先把prediction定义为全局变量
    y_pre = sess.run(prediction, feed_dict={xs:v_xs,on_train : False}) #生成预测值（概率），10分类，所以一个样本是10列概率
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1)) #比较概率最大值的位置和真实标签位置是否一样，一样就是true
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32)) #计算平均正确率
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys}) #生成正确率
    return result



# 只要定义为placeholder的都需要feed传入    
    
with tf.name_scope("inputs"):
    # 输入数据：None不限制样本数，784=28x28,每个样本784个像素点（特征）
    xs = tf.placeholder(tf.float32, [None,inputs_size*inputs_size], name = "x_input") 
    # 输出数据：None不限制样本数，10个输出（10分类问题）
    ys = tf.placeholder(tf.float32, [None,output_size], name = "y_input") 
    # cnn中需要将输入数据reshape为图片形式
    x_image = tf.reshape(xs,[-1,inputs_size,inputs_size,input_channel]) # -1:不考虑样本数，28,28:784像素=28x28,1：一个通道，黑白图片
    #print(x_image.shape)

on_train = tf.placeholder(tf.bool,name="on_train")

# c1:6个5x5的卷积核,SAME padding，得到边长 = 28x28x6
with tf.name_scope("conv_1"):
    conv1 = tf.layers.conv2d(inputs = x_image,
                             filters=filters_1,
                             kernel_size=kernel_size,
                             strides=conv_strides,
                             padding="same",
                             use_bias = False,
                             activation=None)
    conv1 = tf.layers.batch_normalization(conv1,training=on_train)
    conv1 = tf.nn.relu(conv1)
temp_size = inputs_size+4-5+1

# s2:pool核2x2，步长2，28x28x6变成14x14x6
pool2 = tf.layers.max_pooling2d(conv1, pool_size=pool_size, strides=pool_strides, name="pool_2")
temp_size = int(temp_size/2)

# c3:16个5x5的卷积核，得到变成14-5+1 = 10x10x16
with tf.name_scope("conv_3"):
    conv3 = tf.layers.conv2d(inputs= pool2,
                             filters=filters_2,
                             kernel_size=kernel_size,
                             strides=conv_strides,
                             padding="valid",
                             use_bias = False,
                             activation=None)
    conv3 = tf.layers.batch_normalization(conv3,training=on_train)
    conv3 = tf.nn.relu(conv3)
temp_size = temp_size-5+1

# s4:pool核2x2，步长2，10x10x16变成5x5x16
pool4 = tf.layers.max_pooling2d(conv3, pool_size=pool_size, strides=pool_strides, name = "pool_4")
temp_size = int(temp_size/2)

# c5:120个5x5卷积核，得到变成5-5+1 = 1x1x120
with tf.name_scope("conv_3"):
    conv5 = tf.layers.conv2d(inputs= pool4,
                             filters=filters_3,
                             kernel_size=kernel_size,
                             strides=conv_strides,
                             padding="valid",
                             use_bias = False,
                             activation=None)
    conv5 = tf.layers.batch_normalization(conv5,training=on_train)
    conv5 = tf.nn.relu(conv5)
temp_size = (temp_size-5+1)*filters_3

##flat
flat = tf.reshape(conv5,[-1,1*1*temp_size])

with tf.name_scope("fc_6"):
    fc6 = tf.layers.dense(inputs=flat,
                          units=fc_units,
                          use_bias = False,
                          activation=None)
    fc6 = tf.layers.batch_normalization(fc6,training=on_train)
    fc6 = tf.nn.relu(fc6)

fc6 = tf.layers.dropout(inputs=fc6,
                  rate = drop_prob,
                  training = on_train,name='dropout')


prediction = tf.layers.dense(inputs=fc6,
                      units=output_size,
                      activation=tf.nn.softmax,name="prediction")

    
with tf.name_scope("cross_entropy"):
    cross_entropy =  tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=prediction)
    tf.summary.scalar("cross_entropy",cross_entropy)
    
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    with tf.name_scope("train"):
        train_step = tf. train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        
sess = tf.Session()

# 将上述可视化合并
merged = tf.summary.merge_all()

# 将以上结构写入文件，分为train和test
train_writer = tf.summary.FileWriter("/home/kenn/tensorflow_learning/logs/LeNet_5/train",sess.graph)
test_writer = tf.summary.FileWriter("/home/kenn/tensorflow_learning/logs/LeNet_5/test",sess.graph)
# 终端激活对应环境后执行 tensorboard --logdi'file:///home/kenn/tensorflow_learning/logs'
# localhost:6006看

init = tf.global_variables_initializer()
sess.run(init)
for i in range(5000):
    batch_xs,batch_ys = mnist.train.next_batch(128)
    #通过placeholder定义输入的话，都要用feed_dict载入数据，dropout参数也要从这里输入
    sess.run(train_step, feed_dict = {xs:batch_xs, ys:batch_ys, on_train:True})
    
    if i%100 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
        
        #预测的时候droprate为0
        train_result = sess.run(merged, feed_dict = {xs:batch_xs, ys:batch_ys, on_train:False})
        test_result = sess.run(merged, feed_dict = {xs:mnist.test.images, ys:mnist.test.labels, on_train:False})
        
        train_writer.add_summary(train_result,i)
        test_writer.add_summary(test_result,i)
