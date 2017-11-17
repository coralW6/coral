#!/usr/bin/env python
# coding=utf8
# brief: 评论使用卷积神经网络做标签挖掘
# auth: wanglei
# date: 2017-10-13

# 参考:http://www.jianshu.com/p/ed3eac3dcb39文章思想

import sys
import os
reload(sys)
sys.setdefaultencoding("utf-8")

import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet

#One-Hot编码，又称为一位有效编码，主要是采用位状态寄存器来对个状态进行编码，每个状态都由他独立的寄存器位，并且在任意时候只有一位有效。

#数据一共是22w条,每一条是3845列,第一列是结果tag,后面的3844列是训练集数据,后面需要转成62*62的矩阵训练
train_data_file = 'train_200*20_50000.csv'
#test_data_file = 'data/test.csv'

train_data = pd.read_csv(train_data_file).as_matrix().astype(np.float64)

def extract_images_and_labels(dataset):
    ##需要将数据转化为[image_num, x, y, depth]格式
    images = dataset[:, 1:].reshape(-1, 20, 200, 1)  #将每一行的数据转化成20*200的矩阵

    #由于label为0~9,将其转化为一个向量.如将0 转换为 [1,0,0,0,0,0,0,0,0,0]
    labels_dense = dataset[:, 0]
    labels_dense_list = []
    for i in dataset[:, 0]:
        labels_dense_list.append(int(i))     #将float标签转换成int类型
    labels_dense = np.array(labels_dense_list)
    num_labels = labels_dense.shape[0]   #获取行数
    index_offset = np.arange(num_labels) * 10  #给每一行的下标*10
    labels_one_hot = np.zeros((num_labels, 10))  #创建一个n行 * 10的矩阵, 值是0
    #print "index_offset + labels_dense.ravel():", index_offset + labels_dense.ravel()
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1  #将源数据中第i行的输出值v映射到矩阵的第i行第v列的值为1,其余值为0 (one-hot编码)

    num_images = images.shape[0]
    divider = num_images - 500  #分开训练集和验证集
    return images[:divider], labels_one_hot[:divider], images[divider+1:], labels_one_hot[divider+1:]


#创建一个卷积核权重的节点
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)  #shape表示生成张量的维度, stddev是标准差
    return tf.Variable(initial)

#创建一个偏置量的节点 初始值为0.1
def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

"""TensorFlow在卷积和池化上有很强的灵活性。我们怎么处理边界？步长应该设多大?
我们的卷积使用1步长（stride size），0边距（padding size）的模板."""
def conv2d(x,W):
    print "222 conv2d"
    return tf.nn.conv2d(x,W,strides=[1,1,200,1],padding='SAME')   #padding=SAME,采用的是补全的方式,对于上述的情况,允许滑动3次,但是需要补3个元素,左奇右偶,在左边补一个0,右边补2个0

"""我们的池化用简单传统的2x2大小的模板做max pooling。为了代码更简洁，我们把这部分抽象成一个函数。"""
def max_pool_2x2(x):
    print "333 max_pool_2x2"
    return tf.nn.max_pool(x,ksize=[1,20,1,1], strides=[1,1,1,1],padding='SAME')

"""
    测试数据时feed全部数据可能会造成Out of memory 异常，这里将test集拆解为mini-batch
"""
def get_batchs(data, batch_size):
    size = data.shape[0]
    for i in range(size//batch_size):
        if (i+1)*batch_size > size:
            yield data[i*batch_size:]
        else:
            yield data[i*batch_size:(i+1)*batch_size]

if __name__ == '__main__':
    train_images, train_labels, val_images, val_labels = extract_images_and_labels(train_data)
    #test_images = extract_images(test_data)
    #将训练集/验证集和测试集的数据保存起来, dataset可以直接生成mini-batch便于后续训练数据集
    train = DataSet(train_images, train_labels, dtype=np.float32, reshape=True)
    validation = DataSet(val_images, val_labels, dtype=np.float32, reshape=True)
    #test = test_images

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
    W_conv1_1 = weight_variable([2,200,1,64])
    b_conv1_1 = bias_variable([64])


    """我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。"""
    h_conv1_1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1_1, strides=[1,1,1,1],padding='VALID') + b_conv1_1)
    h_pool1_1 = tf.nn.max_pool(h_conv1_1,ksize=[1,5,1,1], strides=[1,5,1,1],padding='SAME')

    #第一层的第二个卷积核
    W_conv1_2 = weight_variable([3,200,1,64])
    b_conv1_2 = bias_variable([64])


    """我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。"""
    h_conv1_2 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1_2, strides=[1,1,1,1],padding='VALID') + b_conv1_2)
    h_pool1_2 = tf.nn.max_pool(h_conv1_2,ksize=[1,6,1,1], strides=[1,6,1,1],padding='SAME')

    #第一层的第三个卷积核
    W_conv1_3 = weight_variable([4,200,1,64])
    b_conv1_3 = bias_variable([64])

    """我们把x_image和权值向量进行卷积，加上偏置项，然后应用ReLU激活函数，最后进行max pooling。"""
    h_conv1_3 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1_3, strides=[1,1,1,1],padding='VALID') + b_conv1_3)
    h_pool1_3 = tf.nn.max_pool(h_conv1_3,ksize=[1,6,1,1], strides=[1,6,1,1],padding='SAME')

    h_pool1 = tf.concat([h_pool1_1, h_pool1_2, h_pool1_3], 1)

    #第二层
    """为了构建一个更深的网络，我们会把几个类似的层堆叠起来。第二层中，每个5x5的patch会得到64个特征。"""
    W_conv2 = weight_variable([2,1,64,128])
    b_conv2 = bias_variable([128])

    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv2, strides=[1,1,1,1],padding='SAME') + b_conv2)
    h_pool2 = tf.nn.max_pool(h_conv2,ksize=[1,2,1,1], strides=[1,2,1,1],padding='SAME')

    """
    全连接层
    现在，图片尺寸减小到16*16，我们加入一个有1024个神经元的全连接层，用于处理整个图片。我们把池化层输出的张量reshape成一些向量，乘上权重矩阵，加上偏置，然后对其使用ReLU。
    """

    W_fc1 = weight_variable([5*1*128, 1024])
    b_fc1 = bias_variable([1024])

    h_pool4_flat = tf.reshape(h_pool2, [-1, 5*1*128])
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
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2   #矩阵相乘

    """
    为了进行训练和评估，在feed_dict中加入额外的参数keep_prob来控制dropout比例。然后每100次迭代输出一次日志。
    """
    cross_entropy = tf.reduce_mean(  #取平均值
        tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv)  #计算最后一层是softmax层的cross entropy(交叉熵) 输入是y_conv, 输出是y_  用来衡量训练标记与真实标记的相似性
    )

    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  #寻找全局最优点的优化算法,计算出梯度,将梯度作用在变量上

    #tf.equal对比两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
    correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))  #tf.argmax对矩阵按行或列计算最大值  0表示按列，1表示按行
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #tf.cast 将数据格式转化成dtype  取平均值

    # 模型持久化
    saver = tf.train.Saver(tf.global_variables())

    sess = tf.InteractiveSession()  #加载它自身作为默认构建的session
    sess.run(tf.global_variables_initializer())   #添加用于初始化变量的节点
    print '111 开始训练'
    for i in range(10000):
        batch = train.next_batch(200)   #每一步迭代，加载50个训练样本
        #loss_ret = train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        loss_ret = sess.run([cross_entropy, train_step], feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
        if i%50 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                x:batch[0], y_: batch[1], keep_prob: 1.0})

            test_accuracy = accuracy.eval(feed_dict={
                x: validation.images, y_: validation.labels, keep_prob: 1.0})
            print "step %d, 测试集的准确率 %g, 验证集的准确率 %g"%(i, train_accuracy, test_accuracy), "loss:", loss_ret

            # 模型持久化
            saver.save(sess, './model_test02/comment_tag_model')

    #使用验证集验证
    print("准确率: %g"%accuracy.eval(feed_dict={
        x: validation.images, y_: validation.labels, keep_prob: 1.0}))
