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

def analysis():
    count = 0
    for line in sys.stdin:
        count += 1

    print count

def main():
    analysis()
if __name__ == '__main__':
    main()
