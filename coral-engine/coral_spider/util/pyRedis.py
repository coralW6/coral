# -*- coding: utf-8 -*-
# brief redis链接和配置
# author wanglei
# date 2017-07-14

import redis
import sys
import os

class Redis(object):
    def __init__(self):
        conf_dict = {
            'host': '127.0.0.1',
            'port': 6379,
            'password': '111111',
            'db': 1
        }
        self.redis_cli = redis.StrictRedis(**conf_dict)

    def sadd(self, key, value):
        #set add
        return True if self.redis_cli.sadd(key, value) > 0 else False

    def is_exist(self, key, value):
        #if self.redis_cli.exists(key) == 1:
            #判断set中是否包含该value
        return False if self.redis_cli.sismember(key, value) == 0 else True
        # else:
        #     self.sadd(key, '1')
        #     return False

