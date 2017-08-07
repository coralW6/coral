# -*- coding: utf-8 -*-
# brief log输出类
# author wanglei
# date 2017-07-14

import os
import logging

class Logger(object):
    def __init__(self, filename):
        logging.basicConfig(
            level = logging.DEBUG,
            filename = os.getenv('HOME') + '/personalWorkplace/coral-engine/coral_spider/logs/' + filename,
            datefmt = '%Y-%m-%d %H:%M:%S',
            filemode = 'w',
            format='%(asctime)s %(filename)s[line:%(lineno)d] [%(levelname)s] %(message)s',
        )

    def info(self, msg):
        logging.info(msg)

    def debug(self, msg):
        logging.debug(msg)

    def error(self, msg):
        logging.error(msg)
