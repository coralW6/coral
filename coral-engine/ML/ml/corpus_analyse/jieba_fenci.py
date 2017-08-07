#!/usr/bin/python
# -*- coding:utf-8 -*-
# brief jieba将doc中的语料进行分词,并计算词频
# pip install jieba

import sys
import os
import traceback
import jieba
import regex as re

reload(sys)
sys.setdefaultencoding('utf8')

INPUT_DIR = os.getenv('HOME') + '/personalWorkplace/coral-engine/ML/ml/corpus_analyse/data/10'
OUTPUT_DIR = os.getenv('HOME') + '/personalWorkplace/coral-engine/ML/ml/corpus_analyse/data/cut_data'

def analysis(tf_dict, content):
    seg_list = jieba.cut(content, cut_all=False)
    for seg in seg_list:
        seg = ''.join(seg)
        seg = seg.strip()
        seg = re.sub(ur"\p{P}+", '', seg)
        #print seg
        if seg != '' and seg != "\n" and seg != "\n\n":
            tf_dict.setdefault(seg, 0)
            tf_dict[seg] += 1
            # seg = seg.encode('utf8')
            # tf_idf_list.append(seg)

def corpus_cut():
    pathDir =  os.listdir(INPUT_DIR)
    #print pathDir
    num = 0
    for input_file in pathDir:
        tf_dict = {}
        #tf_idf_list = []
        output_file = input_file
        try:
            input_file = INPUT_DIR + '/' + input_file
            with open(input_file, 'r') as fr:
                for content in fr:
                    try:
                        #print content
                        analysis(tf_dict, content)
                    except:
                        traceback.print_exc()
                fr.close()
        except:
            traceback.print_exc()

        #输出
        output_file = output_file.replace('.txt', '')
        #print output_file
        with open('%s/%s.txt' % (OUTPUT_DIR, output_file), 'w') as fw:
            for seg in tf_dict.keys():
                count = tf_dict[seg]
                seg = seg.encode('utf8')
                fw.writelines([seg, ':', str(count), '\n'])
            fw.close()

        num += 1
        print num
        # if num >= 1000:
        #     break

def main():
    corpus_cut()
if __name__ == '__main__':
    main()
