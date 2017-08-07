#!/usr/bin/python
# -*- coding:utf-8 -*-

import pandas as pd  #分析数据
import numpy as np #计算
import sklearn  #算法模型
import matplotlib.pyplot as plt #画图

def analyse(data_train):
    # plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
    # plt.rcParams['font.family']='sans-serif'
    # plt.rcParams['axes.unicode_minus'] = False # 用来正常显示负号
    # fig = plt.figure()
    # fig.set(alpha=0.2) # 设定图表颜色alpha参数

    # plt.subplot2grid((2,3),(0,0)) # 在一张大图里分列几个小图
    # data_train.Survived.value_counts().plot(kind='bar')# 柱状图
    # plt.title(u"获救情况 (1为获救)") # 标题
    # plt.ylabel(u"人数") # Y轴标签
    # plt.show()
    #
    # plt.subplot2grid((2,3),(0,1)) # 在一张大图里分列几个小图
    # data_train.Pclass.value_counts().plot(kind='bar')# 柱状图
    # plt.title(u"乘客等级分布") # 标题
    # plt.ylabel(u"人数") # Y轴标签
    # plt.show()
    #
    # #等级和获救有关
    # plt.subplot2grid((2,3),(1,0)) # 在一张大图里分列几个小图
    # Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts() # 未获救
    # Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts() # 获救
    # df = pd.DataFrame({'获救': Survived_1, '未获救': Survived_0})
    # df.plot(kind = 'bar', stacked = True)
    # plt.title(u"乘客等级和获救人数分布") # 标题
    # plt.ylabel(u"等级") # Y轴标签
    # plt.ylabel(u"人数") # Y轴标签
    # plt.show()

    #性别和获救有关
    # plt.subplot2grid((2,3),(1,0)) # 在一张大图里分列几个小图
    # Survived_0 = data_train.Survived[data_train.Sex == 'male'].value_counts() # 男
    # Survived_1 = data_train.Survived[data_train.Sex == 'female'].value_counts() # 女
    # df = pd.DataFrame({'男': Survived_0, '女': Survived_1})
    # df.plot(kind = 'bar', stacked = True)
    # plt.title(u"性别和获救人数分布") # 标题
    # plt.ylabel(u"等级") # Y轴标签
    # plt.ylabel(u"人数") # Y轴标签
    # plt.show()


    #高级仓女性生还率很高,低级仓男性生还率很低
    # fig = plt.figure()
    # plt.title(u'根据舱等级和性别的获救情况')
    #
    # ax1 = fig.add_subplot(141) # 将图像分为1行4列，从左到右从上到下的第1块
    # data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind = 'bar', label = 'female high class', color = '#FA2479')
    # ax1.set_xticklabels([u'获救',u'未获救'], rotation = 0) # 根据实际填写标签
    # ax1.legend([u'女性/高级舱'], loc = 'best')
    #
    # ax2 = fig.add_subplot(142, sharey = ax1) # 将图像分为1行4列，从左到右从上到下的第2块
    # data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind = 'bar', label = 'female low class', color = 'pink')
    # ax2.set_xticklabels([u"未获救", u"获救"], rotation=0)
    # plt.legend([u"女性/低级舱"], loc='best')
    #
    # ax3 = fig.add_subplot(143, sharey = ax1)
    # data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind = 'bar', label = 'male high class', color = 'lightblue')
    # ax3.set_xticklabels([u'未获救',u'获救'], rotation = 0)
    # plt.legend([u'男性/高级舱'], loc = 'best')
    #
    # ax4 = fig.add_subplot(144, sharey = ax1)
    # data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind = 'bar', label = 'male low class', color = 'steelblue')
    # ax4.set_xticklabels([u'未获救',u'获救'], rotation = 0)
    # plt.legend([u'男性/低级舱'], loc = 'bast')
    # plt.show()


    dumm = pd.get_dummies(data_train[['Sex','Embarked']]) # '哑变量'矩阵
    data_train = data_train.join(dumm) #将哑变量矩阵和原有的矩阵合并
    print data_train


def main():
    data_train = pd.read_csv('train.csv')
    #print data_train.info()
    print data_train.describe()
    #analyse(data_train)

    data_test = pd.read_csv('test.csv')
    print data_test.info()
    data_test.to_csv()


if __name__ == '__main__':
    main()
