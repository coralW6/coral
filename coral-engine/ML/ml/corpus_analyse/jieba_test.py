#!/usr/bin/python
# -*- coding:utf-8 -*-
# brief jieba的使用
# pip install jieba

import jieba
import os
import sys
import traceback
import re

# s = """
#     施工方的武研项目管理项目管理的核心任务项目的目标控制根据《建设工程质量管理条例》，监理工程师应当按照工程监理规范的要求，采取等形式对建设工程实施监理
# """
#
# tmp =  jieba.cut(s, cut_all=True)
# print "/".join(tmp)
# tmp =  jieba.cut(s, cut_all=False)
# print "/".join(tmp)

INPUT_DIR = os.getenv('HOME') + '/personalWorkplace/coral-engine/ML/ml/corpus_analyse/data/10'
OUTPUT_DIR = os.getenv('HOME') + '/personalWorkplace/coral-engine/ML/ml/corpus_analyse/data/cut_data'


pathDir =  os.listdir(INPUT_DIR)

#print pathDir
def read_file(count):
    for input_file in pathDir:
        tf_dict = {}
        output_file = input_file
        try:
            input_file = INPUT_DIR + '/' + input_file
            with open(input_file, 'r') as fr:
                for content in fr:
                    try:
                        print 'bbbbbbbbbbbbbbb', count
                        yield content
                    except:
                        traceback.print_exc()
                fr.close()
        except:
            traceback.print_exc()

def main():
    count = 0
    print '>>>start'
    data = read_file(count)
    print 'qqq', count
    count += 1
    for seg in data:
        seg = re.sub(ur"\p{P}+", '', seg)
        print 'aaaa'
        print seg


if __name__ == '__main__':
    main()