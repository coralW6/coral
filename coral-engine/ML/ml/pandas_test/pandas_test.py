#!/usr/bin/python
# -*- coding:utf-8 -*-
# brief pandas的使用
# pip install pandas

import pandas as pd
import numpy as np
import plistlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression # 引入逻辑回归
from sklearn.cross_validation import train_test_split


df = pd.read_excel('test_data.xls')
print type(df)
#print df, '\n'
#print df.sort_index()
#print df.sort_values(['col_a', 'col_c'])
#print df.sort(['col_a'])
#df.dropna(how='any')
#print df.mean()
print df.apply(lambda x : x.max()- x.min())
df = df.cumsum()
plt.figure()
df.plot()
plt.legend(loc='best')
LR = LogisticRegression()
train_test_split()


