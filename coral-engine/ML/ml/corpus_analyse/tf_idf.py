#!/usr/bin/python
# -*- coding:utf-8 -*-
# brief 计算tf-idf

import sys
import os
import jieba
import re

reload(sys)
sys.setdefaultencoding('utf8')

#INPUT_DIR = os.getenv('HOME') + '/personalWorkplace/coral-engine/ML/ml/corpus_analyse/data/idf.txt'
INPUT_DIR = 'idf.txt'
word_idf = {}

def init():
    with open(INPUT_DIR, 'r') as fr:
        for line in fr:
            if line.strip():
                word = line.split(' ')[0]
                idf_val = line.split(' ')[1]
                word_idf[word.encode('utf8')] = idf_val
                #print word, idf_val
def main(data):
    word_count = {}
    seg_list = jieba.cut(data, cut_all=False)
    for seg in seg_list:
        seg = ''.join(seg)
        seg = seg.strip()
        seg = re.sub(ur"\p{P}+", '', seg)
        if seg != '' and seg != "\n" and seg != "\n\n":
            word_count.setdefault(seg, 0)
            word_count[seg] += 1

    print word_count
    ret = {}
    for word in word_count:
        #print word
        count = word_count[word]
        if word_idf.has_key(word):
            print word
            idf_val = word_idf[word]
            tf_idf = count * idf_val
            ret[word] = tf_idf
            #print word, tf_idf

    print ret
    out_list  = sorted(ret.items(), key=lambda d: d[1], reverse=True)
    #print out_list
    out_list = out_list[0:20]
    # for r in out_list:
    #     print r[0], r[1]

if __name__ == '__main__':
    #if len(sys.stdin) >= 0:
    text = ""
    for t in  sys.stdin.readlines():
        text += t.encode('utf8')
    print text
    init()
    main(text)