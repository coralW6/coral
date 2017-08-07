#!/usr/bin/python
# -*- coding:utf-8 -*-
# brief jieba将doc中的语料进行分词,提取单词
# pip install jieba

import sys
import os
sys.path.append(os.getcwd())

import traceback
import jieba
import re

reload(sys)
sys.setdefaultencoding('utf8')

def read_data(file):
    for line in file:
        yield line


def analysis():
    word_set = set()
    data_list = read_data(sys.stdin)
    for data in data_list:
        seg_list = jieba.cut(data, cut_all=False)
        for seg in seg_list:
            seg = ''.join(seg)
            seg = seg.strip()
            seg = re.sub(ur"\p{P}+", '', seg)
            if seg != '' and seg != "\n" and seg != "\n\n":
                word_set.add(seg)
                # tf_dict.setdefault(seg, 0)
                # tf_dict[seg] += 1

    for word in word_set:
        print "%s:%s" % (word, 1)


def main():
    analysis()
if __name__ == '__main__':
    main()
