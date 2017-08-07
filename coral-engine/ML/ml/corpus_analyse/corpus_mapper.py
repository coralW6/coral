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

reload(sys)
sys.setdefaultencoding('utf8')

INPUT_DIR = os.getenv('HOME') + '/personalWorkplace/coral-engine/ML/ml/corpus_analyse/data/cut_data'
OUTPUT_DIR = os.getenv('HOME') + '/personalWorkplace/coral-engine/ML/ml/corpus_analyse/data/idf_data'

files = os.listdir(INPUT_DIR)

word_dict = {}
for self_file in files:
    self_file = INPUT_DIR +'/' + self_file
    with open(self_file, 'r') as fr:
        for line in fr:
            word = line.split(':')[0]
            word = word.encode('utf8')
            word_dict.setdefault(word, 0)
            word_dict[word] += 1

with open('%s/word_idf.txt' % OUTPUT_DIR, 'w') as fw:
    for r in word_dict.keys():
        count = word_dict[r]
        fq = math.log((3000/count))
        print fq
        fw.writelines([r, ':', str(word_dict[r]), ':', str(fq), '\n'])
    fw.close()

