#!/usr/bin/python
# -*- coding:utf-8 -*-
# brief 统计包含某个词的文档个数

#0 将原始文doc切词后以{word:该文档中出现的次数(tf)}存放起来
#1 写map计算出来某个词出现在多少个doc中,建立 {Word:出现的doc数} 存放起来
#2 写reduce计算idf = log(总doc数/出现该词的doc数),将值存放在目录下
#3 写reduce计算每个doc中各词的权重 = tf * idf,取其中的前5,表示该doc的关键词

import os
import sys
import math
#import jieba

reload(sys)
sys.setdefaultencoding('utf8')

CUT_DIR = os.getenv('HOME') + '/personalWorkplace/coral-engine/ML/ml/corpus_analyse/data/cut_data'
INPUT_DIR = os.getenv('HOME') + '/personalWorkplace/coral-engine/ML/ml/corpus_analyse/data/idf_data'
OUTPUT_DIR = os.getenv('HOME') + '/personalWorkplace/coral-engine/ML/ml/corpus_analyse/data/keyword_data'

files = os.listdir(INPUT_DIR)
word_weight_dict = {}

for self_file in files:
    self_file = INPUT_DIR + '/' + self_file
    with open(self_file, 'r') as fr:
        for line in fr:
            word = line.split(':')[0]
            weight = line.split(':')[2]
            word_weight_dict[word] = weight


cur_files = os.listdir(CUT_DIR)
for self_file in cur_files:
    self_file = CUT_DIR + '/' + self_file
    with open(self_file, 'r') as fr:
        out_dict = {}
        for line in fr:
            word = line.split(':')[0]
            count = line.split(':')[1].replace('\n', '')
            if word_weight_dict.has_key(word):
                weight = word_weight_dict[word].replace('\n', '')
                word_weight = int(count) * float(weight)
                out_dict[word] = word_weight
    fr.close()
    out_list = sorted(out_dict.items(), key=lambda d: d[1], reverse=True)
    out_list = out_list[0:20]
    for r in out_list:
        print r[0], r[1]
    break


