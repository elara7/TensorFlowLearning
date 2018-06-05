#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 10:42:52 2018

@author: kenn
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as  plt

pi_rate = 0.25
pi_int = int(10/pi_rate)
BATCH_START = 0
TIME_STEPS = 20 #每个样本包含的时间步
BATCH_SIZE = pi_int
NUM_FEATURES = 1 #每个时间步中包含的特征数
OUTPUT_SIZE = 1 #输出序列中每个时间步的特征数
CELL_SIZE = 32 #RNN、LSTM cell的单元数
LR = 0.006
BATCH_START_TEST = 0


def get_batch():
    global BATCH_START, TIME_STEPS
    # xs.shape = [batch, 20 steps],每行一个样本，每个样本20步
    xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS*BATCH_SIZE).reshape([BATCH_SIZE,TIME_STEPS])/(TIME_STEPS)*pi_rate*np.pi    
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS # 滑动一个样本
    return [seq[:,:,np.newaxis], res[:,:,np.newaxis],xs]

x = tf.placeholder(tf.float32, [None, TIME_STEPS, NUM_FEATURES])       
y = tf.placeholder(tf.float32, [None, TIME_STEPS, NUM_FEATURES])                  





# 预建基础单元#################################################################
# 一批样本的第一步：[batch,num_features]，因为只有一步，所以没有time_step维度
# 生成隐状态[batch,cell_size]
# 生成该步的输出：[batch，cell_size]
cell = tf.contrib.rnn.BasicLSTMCell(num_units=CELL_SIZE)

# 生成初始化的隐状态###########################################################
state = cell.zero_state(BATCH_SIZE, tf.float32)
# LSTM的状态有2个，需要分别untrainable化再打包
state = tf.contrib.rnn.LSTMStateTuple(
            tf.Variable(state[0], trainable=False),
            tf.Variable(state[1], trainable=False))

# dynamic_rnn自动执行lstm_cell的每一步#########################################
# 输入一批样本：[batch,time_step,num_features]
# 输出outputs为所有步的结果：[batch,time_step,num_features*cell_size]
# new_state = (c_state, m_state)为最终得到的隐含结果
outputs, new_state = tf.nn.dynamic_rnn(cell,inputs= x,
                                       initial_state= state, #如果为None，且有指定dtype，则自动生成
                                       dtype=tf.float32, 
                                       time_major=False # 第一维不是time_step
                                       )

# 设定控制依赖，方便不同batch继承##############################################
with tf.control_dependencies([state[0].assign(new_state[0]), state[1].assign(new_state[1])]):
    outputs = tf.identity(outputs)

# RNN 版本，其他单state的模型通用##############################################
#cell = tf.contrib.rnn.BasicRNNCell(num_units=CELL_SIZE)
#state = cell.zero_state(BATCH_SIZE, tf.float32)
#state = tf.Variable(state, trainable=False)
#outputs, new_state = tf.nn.dynamic_rnn(cell,inputs= x,
#                                       initial_state= state, #如果为None，且有指定dtype，则自动生成
#                                       dtype=tf.float32, 
#                                       time_major=False # 第一维不是time_step
#                                       )
#with tf.control_dependencies([state.assign(new_state)]):
#    outputs = tf.identity(outputs)
    

# 将cell输出的隐层状态用fc连接到最终结果（隐藏单元数->输出单元数）###############
# reshape 3D output to 2D for fully connected layer
outs2D = tf.reshape(outputs, [-1, CELL_SIZE])                       
net_outs2D = tf.layers.dense(outs2D, NUM_FEATURES)
# reshape back to 3D
outs = tf.reshape(net_outs2D, [-1, TIME_STEPS, NUM_FEATURES])          

# compute cost
cost = tf.losses.mean_squared_error(labels=y, predictions=outs)  
train_op = tf.train.AdamOptimizer(LR).minimize(cost)

mse = tf.metrics.mean_squared_error(labels=y, predictions=outs)

sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("/home/kenn/tensorflow_learning/logs/RNN_reg/",sess.graph)

sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])


plt.ion()
plt.show()

for i in range(1,300):
    seq,res,xs = get_batch()
    
    plt.plot(xs[:pi_int].flatten(), res[:pi_int].flatten(),'r',
             xs[:pi_int].flatten(), seq[:pi_int].flatten(),'b--')
    
    feed_dict = {x:seq, y:res}

    _,mse_,pred = sess.run([train_op,mse,outs
                                    ],feed_dict=feed_dict)
    
    
    plt.clf()
    plt.cla()
    plt.plot(xs[:pi_int].flatten(), res[:pi_int].flatten(),'r',
             xs[:pi_int].flatten(), pred[:pi_int].flatten(),'b--')
    plt.ylim((-1.2,1.2))
    plt.draw()
    plt.pause(0.05)
    
    
    if i%10 ==0:
        print('mse',mse_[0])

