#!/usr/bin/python
# -*- coding:utf-8 -*-
# brief 计算词频

import sys
import os
import math
sys.path.append(os.getcwd())

import traceback

reload(sys)
sys.setdefaultencoding('utf8')

def main():
    word_count = {}
    #total_count = 2050515
    total_count = 3000
    for line in sys.stdin:
        #total_count += 1
        word = line.split(':')[0]
        count = line.split(':')[1]
        if count.isdigit():
            count = int(count)
        else:
            count = 1
        word_count.setdefault(word, 0)
        word_count[word] += count

    for word in word_count.keys():
        idf_val = math.log((total_count / word_count[word]))
        #print word + ':' + word_count[word] + ':' + idf_val + ': \n'
        print "%s:%s:%s\n" % (word, word_count[word], idf_val)

if __name__ == '__main__':
    main()
