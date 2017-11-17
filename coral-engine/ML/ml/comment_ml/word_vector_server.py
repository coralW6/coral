#!/usr/bin/env python
# -*- coding: utf-8 -*-
#author wanglei
#brief 词向量转换服务
#date 2017-10-14

import sys
from BaseHTTPServer import HTTPServer, BaseHTTPRequestHandler
import json
import time
import traceback
import struct
import urllib

reload(sys)
sys.setdefaultencoding('utf-8')


addr = ('127.0.0.1', 6666)

word_vec_path = '/Users/bjhl/personalWorkplace/coral/coral-engine/ML/ml/result/vectors.bin.3g200dim'
#word_vec_path = 'vectors.bin.3g200dim'

stop_word_set = set()
word_vector_dict = {}

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

class MyService(BaseHTTPRequestHandler):
    #172.16.1.10:6666?word=中国
    def do_GET(self):
        try:
            print 'get1111111', self.path
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            params = self.parseGetRequest(self.path)
            #print '>>>>params:', params
            if params:
                #print params
                ret = self.result(params)
                print 'ret_size:', len(ret)
                self.wfile.write(json.dumps(ret))
        except:
            print 'GET error'
            traceback.print_exc()
            exit()

    def do_POST(self):
        pass


    def result(self, params):
        ret_list = []
        word = params["word"]
        word = urllib.unquote(word)
        word = unicode(word.strip())
        print "word:", word
        if len(word) <= 0:
            return ret_list
        if word in stop_word_set:
            return ret_list
        if not word.isalpha():
            return ret_list
        if word_vector_dict.has_key(word):
            ret_list = word_vector_dict[word]
        return ret_list

    def parseGetRequest(self, path):
        if path is not None and '?' in self.path:
            params_dict = {}
            params = self.path.split('?')[1]
            for query in params.split('&'):
                query_key = query.split('=')[0]
                query_value = query.split('=')[1]
                params_dict[query_key] = query_value
            return params_dict
        return None

def main():
    try:
        load_stop_word()
        init_word_vec()
        myService = HTTPServer(addr, MyService)
        print '>begin listen<'
        myService.serve_forever()
    except:
        print 'myService error'
        traceback.print_exc()

if __name__ == '__main__':
    try:
        main()
    except:
        print 'main error'
        traceback.print_exc()

