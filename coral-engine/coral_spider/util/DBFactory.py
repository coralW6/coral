#!/usr/bin/python
# -*- coding: utf-8 -*-
# brief 数据库连接池
# author wanglei
# date 2017-07-16

import os
from Properties import Properties
from DBUtil import DBUtil

class DBFactory(object):
    def __init__(self):
        self.dbs = {}
        home = os.getenv('HOME')
        pro = Properties(home + '/personalWorkplace/coral/coral-engine/coral_spider/resources/db.properties')
        self.property = pro.get_properties()

    def get_db(self, name):
        """
        :rtype: DBUtil
        """
        charset = 'utf8'
        if name in self.dbs:
            return self.dbs[name]
        else:
            # 读取数据库配置
            read_conf = {
                'host': self.property[name + '.host'],
                'user': self.property[name + '.user'],
                'passwd': self.property[name + '.passwd'],
                'db': self.property[name + '.db'],
                'port': int(self.property[name + '.port']),
                'charset': charset,
                'connect_timeout': 30
            }
            write_conf = read_conf
            self.dbs[name] = DBUtil(read_conf, read_conf)
            self.dbs[name] = DBUtil(read_conf, write_conf)
        return self.dbs[name]


factory = DBFactory()

if __name__ == '__main__':
    db = factory.get_db('coral')
    print db.fetch_data_with_map('select 1')
