#!/usr/bin/env python
# coding=utf8
# brief: 预测评论的标签
# auth: wanglei
# date: 2017-10-13

import sys
import os
import tensorflow as tf
import pandas as pd
import numpy as np
import jieba
import traceback
import urllib
import urllib2
import json
import urllib

from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

reload(sys)
sys.setdefaultencoding("utf-8")

requestUrl = "http://127.0.0.1:6666?"

tag_map = {
    1: "内容新颖有用",
    2: "颜值高",
    3: "态度认真负责",
    4: "幽默风趣",
    5: "善于沟通",
    6: "性价比高",
    7: "上课准时",
    8: "课堂气氛活跃",
    9: "授课熟练",
}

total_word = 20
total_data_len = 4000

def send_word(word):
    try:
        params=urllib.urlencode({'word': word})
        response = urllib2.urlopen(requestUrl + params, timeout=60)
        ret = response.read()
        json_obj = json.loads(ret)
        #print "ret:", json_obj
        return json_obj
    except Exception as e:
        print 'get clues params error: %s' % params
        print traceback.print_exc()

def extends_zero(content_vec):
    data_len = len(content_vec)
    zero_len = total_data_len - data_len
    if zero_len > 0:
        for i in range(0, zero_len):
            content_vec.append(0)

def parse(content):
    if content and len(content) > 0:
        content = str(content)
        seg_list = jieba.cut(content)
        #print content
        count = 0
        content_vec = []
        for word in seg_list:
            if count < total_word:
                count += 1
                vec = send_word(word)
                content_vec.extend(vec)
        extends_zero(content_vec)
        return content_vec

#创建一个卷积核权重的节点
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  #shape表示生成张量的维度, stddev是标准差
    return tf.Variable(initial)

#创建一个偏置量的节点 初始值为0.1
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def get_model():
    #定义输入和输出的数据类型和数据形状
    x = tf.placeholder(tf.float32, shape=[None, 4000])  #200列 行不定
    y_ = tf.placeholder(tf.float32, shape=[None, 10])  #10列  行不定

    """为了用这一层，我们把x变成一个4d向量，其第2、第3维对应图片的宽、高，最后一维代表图片的颜色通道数(因为是灰度图所以这里的通道数为1，如果是rgb彩色图，则为3)。"""
    x_image = tf.reshape(x, [-1,20,200,1])   #-1代表的含义是不用我们自己指定这一维的大小，函数会自动计算

    """
    现在我们可以开始实现第一层了。它由一个卷积接一个max pooling完成。
    卷积在每个5x5的patch中算出32个特征。
    卷积的权重张量形状是[5, 5, 1, 32]，前两个维度是patch的大小，接着是输入的通道数目，最后是输出的通道数目。
    而对于每一个输出通道都有一个对应的偏置量。
     """
    #第一层的第一个卷积核
    W_conv1_1 = weight_variable([2,200,1,4096])
    b_conv1_1 = bias_variable([4096])


    """我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。"""
    h_conv1_1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1_1, strides=[1,1,1,1],padding='VALID') + b_conv1_1)
    h_pool1_1 = tf.nn.max_pool(h_conv1_1,ksize=[1,5,1,1], strides=[1,5,1,1],padding='SAME')

    #第一层的第二个卷积核
    W_conv1_2 = weight_variable([3,200,1,4096])
    b_conv1_2 = bias_variable([4096])


    """我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。"""
    h_conv1_2 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1_2, strides=[1,1,1,1],padding='VALID') + b_conv1_2)
    h_pool1_2 = tf.nn.max_pool(h_conv1_2,ksize=[1,6,1,1], strides=[1,6,1,1],padding='SAME')

    #第一层的第三个卷积核
    W_conv1_3 = weight_variable([4,200,1,4096])
    b_conv1_3 = bias_variable([4096])

    """我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。"""
    h_conv1_3 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1_3, strides=[1,1,1,1],padding='VALID') + b_conv1_3)
    h_pool1_3 = tf.nn.max_pool(h_conv1_3,ksize=[1,6,1,1], strides=[1,6,1,1],padding='SAME')

    h_pool1 = tf.concat([h_pool1_1, h_pool1_2, h_pool1_3], 1)

    #第二层
    """为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的patch会得到64个特征。"""
    W_conv2 = weight_variable([2,1,4096,8192])
    b_conv2 = bias_variable([8192])

    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1],padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,2,1,1], strides=[1,2,1,1],padding='SAME')

    """
    全连接层
    现在，图片尺寸减小到16*16，我们加入一个有1024个神经元的全连接层，用于处理整个图片。我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
    """

    W_fc1 = weight_variable([5*1*8192, 4096])
    b_fc1 = bias_variable([4096])

    h_pool4_flat = tf.reshape(h_pool2, [-1, 5*1*8192])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)

    """
    为了减少过拟合，我们在输出层之前加入dropout。我们用一个placeholder来代表一个神经元的输出在dropout中保持不变的概率。
    这样我们可以在训练过程中启用dropout，在测试过程中关闭dropout。
    TensorFlow的tf.nn.dropout操作除了可以屏蔽神经元的输出外，还会自动处理神经元输出值的scale。
    所以用dropout的时候可以不用考虑scale。
    """
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    """
    输出层
    最后，我们添加一个softmax层，就像前面的单层softmax regression一样。
    """
    W_fc2 = weight_variable([4096, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2   #矩阵相乘
    return x, y_conv, keep_prob

def predict(input):
    predict_data = input.reshape(-1, 20*200)
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess, './model_78/comment_tag_model')
        #输出预测的结果
        #outputs_seq = sess.run(outputs, input_feed)
        prediction = tf.argmax(outputs, 1)
        test_labels = prediction.eval(feed_dict={x: predict_data, keep_prob: 1.0})
        for label in test_labels:
            print "label:", label, tag_map[label]

if __name__ == '__main__':
    if len(sys.argv) == 2:
        input_data = parse(sys.argv[1])
        #print "input_data:", input_data, type(input_data)
        global x, outputs, keep_prob
        x, outputs, keep_prob = get_model()
        input = np.array(input_data)
        predict(input)