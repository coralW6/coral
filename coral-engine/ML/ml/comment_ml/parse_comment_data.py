#!/usr/bin/env python
# coding=utf8
# 基础数据处理
"""
    1.将标签映射到对一个的id
    2.将文本切割后转成词向量,最多需要19个词.需要将词向量转成62*62的矩阵,也就是每一行是3844个向量值.不够的补充0
    3.将两者存起来
"""

import os
import sys
import xlwt
import struct
import jieba
import traceback
import csv
reload(sys)
sys.setdefaultencoding("utf-8")

sys.path.append('/Users/bjhl/personalWorkplace/coral/coral-engine/ML/ml/')
word_vec_path = '/Users/bjhl/personalWorkplace/coral/coral-engine/ML/ml/result/vectors.bin.3g200dim'
#word_vec_path = 'vectors.bin.3g200dim'

#将词向量加载到map中
word_vector_dict = {}
#将标签映射成数字
tag_map = {
    "内容新颖有用": 1,
    "颜值高": 2,
    "态度认真负责": 3,
    "幽默风趣": 4,
    "善于沟通": 5,
    "性价比高": 6,
    "上课准时": 7,
    "课堂气氛活跃": 8,
    "授课熟练": 9,
}

total_word = 20
total_data_len = 4001

stop_word_set = set()
def load_stop_word():
    with open("stop_word.txt", "r") as fr:
        for line in fr:
            stop_word_set.add(unicode(line.strip()))

def init_word_vec():
    """加载词向量二进制到内存"""
    float_size = 4  # 一个浮点数4字节
    max_w = 50  # 最大单词字数
    input_file = open(word_vec_path, "rb")
    # 获取词表数目及向量维度
    words_and_size = input_file.readline()
    words_and_size = words_and_size.strip()
    words = long(words_and_size.split(' ')[0])
    word_vec_dim = long(words_and_size.split(' ')[1])
    print("词表总词数：%d" % words)
    print("词向量维度：%d" % word_vec_dim)

    for b in range(0, words):
        a = 0
        word = ''
        # 读取一个词
        while True:
            c = input_file.read(1)
            word = word + c
            if False == c or c == ' ':
                break
            if a < max_w and c != '\n':
                a = a + 1
        word = word.strip()
        vector = []

        for index in range(0, word_vec_dim):
            m = input_file.read(float_size)
            (weight,) = struct.unpack('f', m)
            f_weight = float(weight)
            vector.append(f_weight)

        # 将词及其对应的向量存到dict中
        try:
            #print word.decode('utf-8'), vector[0:word_vec_dim]
            word_vector_dict[word.decode('utf-8')] = vector[0:word_vec_dim]
        except:
            # 异常的词舍弃掉
            print('bad word:' + word)
            pass

    input_file.close()
    print "finish"

def write_list(res, name):
    """
    将数组写入excel
    :param res_list:
    :param name:
    :return:
    """
    if type(res) == list:
        d = {}
        d[name] = res
        res = d
    wb = xlwt.Workbook(encoding='utf-8')
    for sheet_name, res_list in res.items():
        sheet = wb.add_sheet(sheet_name, cell_overwrite_ok=True)
        for i in range(0, len(res_list)):
            for j in range(0, len(res_list[i])):
                sheet.write(i, j, res_list[i][j])

    wb.save('%s.xls' % name)

def write_list_csv(res, name):
    with open(name, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(res)

def deal_tag(tags):
    if tags:
        tag_list = tags.split('|')
        for tag in tag_list:
            if tag_map.has_key(tag):
                return tag_map[tag]
    return None

def deal_content(content):
    if content:
        seg_list = jieba.cut(content)
        #print content
        count = 0
        content_vec = []
        for word in seg_list:
            word = word.strip()
            if len(word) <= 0:
                continue
            if word in stop_word_set:
                continue
            if not word.isalpha():
                #print word
                continue
            #print "valid word:", word
            if count < total_word and word_vector_dict.has_key(word):
                count += 1
                vec = word_vector_dict[word]
                content_vec.extend(vec)
        return content_vec

def extends_zero(base_data):
    data_len = len(base_data)
    zero_len = total_data_len - data_len
    if zero_len > 0:
        for i in range(0, zero_len):
            base_data.append(0)

def do_work():
    res = []
    parse_count = 0
    with open('base_data.txt', 'r') as fr:
        for line in fr:
            parse_count += 1
            if parse_count % 1000 == 0:
                print "解析了 %s 条" % parse_count
            try:
                tags = line.split('$')[0]
                content = line.split('$')[1]
                content_vec = deal_content(content)
                if not content_vec:
                    print "content_vec error", tags, content
                    continue
                tag_list = tags.split('|')
                for tag in tag_list:
                    base_data = []
                    tag_flag = deal_tag(tag)
                    if not tag_flag:
                        continue
                    base_data.append(tag_flag)
                    #print "tag_flag:", tag_flag
                    base_data.extend(content_vec)
                    extends_zero(base_data)
                    res.append((base_data))
            except:
                traceback.print_exc()
        #print "res:", res
        #write_list(res, 'train')
        write_list_csv(res, 'train_200*20.csv')

if __name__ == '__main__':
    load_stop_word()
    init_word_vec()
    do_work()
