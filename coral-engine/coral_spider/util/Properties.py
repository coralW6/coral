#!/usr/bin/python
# -*- coding: utf-8 -*-
# brief 配置
# author wanglei
# date 2017-07-16


class Properties(object):

    def __init__(self, path):
        self.properties = {}
        try:
            p_file = open(path, 'r')
            for line in p_file:
                if line.find('=') > -1:
                    k, v = line.split('=', 1)
                    self.properties[k.strip()] = v.strip()
        except Exception, e:
            print e

    def get_properties(self):
        return self.properties
