# -*- coding: utf-8 -*-
# brief 抓取查字典的小初高作文
# author wanglei
# date 2017-07-17

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from datetime import datetime
from scrapy.http import Request
from scrapy.spider import Spider
from urlparse import urljoin
from pyquery import PyQuery as pq
import logging
import re
from scrapy.utils.project import project_data_dir
import json
import requests
import subprocess
import traceback
import signal
from util.logger import Logger
from util.pyRedis import Redis
from coral_spider.items import ChazidianZuowenItem


class linux_idc(Spider):
    name = 'chazidian_zuowen'
    custom_settings = {
        'ITEM_PIPELINES': {'coral_spider.pipelines.ChazidianZuowenPipeline': 5,}
    }
    def __init__(self):
        try:
            self.redis_key = 'chazidian_zuowen_url'
            self.start_urls = {"xiaoxue": "https://zuowen.chazidian.com/xiaoxuezuowen/",
                               "chuzhong": "https://zuowen.chazidian.com/zhongxuezuowen/",
                               "gaozhong": "https://zuowen.chazidian.com/gaozhongzuowen/"}
            #self.start_urls = {"xiaoxue": "https://zuowen.chazidian.com/xiaoxuezuowen/"}
            self.log = Logger('chazidian_zuowen.log')
            self.redis = Redis()
            self.date = datetime.now().strftime('%Y-%m-%d')
            self.source = 'https://zuowen.chazidian.com'
            self.max_count = 200
        except:
            traceback.print_exc()

    def start_requests(self):
        try:
            self.log.info('开始抓取文章 %s' % self.date)
            for grade in self.start_urls.keys():
                meta = {}
                meta['max_count'] = self.max_count
                meta['grade'] = grade
                yield Request(self.start_urls[grade], callback=self.parse_urls, dont_filter=True, meta=meta)
        except:
            traceback.print_exc()

    def parse_urls(self, response):
        try:
            count = 0
            r = response
            meta = r.meta
            html = r.body_as_unicode()
            html_pq = pq(html)
            article_list = html_pq('div.list_vel3 > ul > li')
            for li_detail in article_list:
                url_pq = pq(li_detail)
                #print url_pq
                a = url_pq('li > a')
                url = a.attr('href')
                category = url_pq('span > a').text()
                title = a.text()
                category = category.replace('[', '')
                category = category.replace(']', '')
                print url,  category,  title
                if self.redis.is_exist(self.redis_key, str(url)):  #去重
                    self.log.info('%s in redis set, continue' % url)
                    continue
                count += 1
                self.log.info('抓取第%s个' % count)
                max_count = meta['max_count']
                if count > max_count:
                    break
                self.redis.sadd(self.redis_key, str(url))
                self.log.info(url)
                meta['title'] = title
                meta['url'] = url
                meta['category'] = category
                yield Request(url, self.parse_detail, meta=meta, dont_filter=True)
            self.log.info('抓取完成,count=%s' % (count-1))

        except:
            traceback.print_exc()

    def parse_detail(self, response):
        try:
            item = ChazidianZuowenItem()
            r = response
            html = r.body_as_unicode()
            html_pq = pq(html)
            meta = r.meta
            author = html_pq('div.cont_mian_art_t_r.fr > p').text()
            author = author.split(' ')[1]
            item['title'] = meta['title']
            item['category'] = meta['category']
            item['grade'] = meta['grade']
            print 'title:', meta['title'], ' category:', item['category'], ' grade:', item['grade']
            item['original_url'] = meta['url']
            content = ''
            content_p_list = html_pq('#print_content > p')
            for p in content_p_list:
                content_detail = pq(p).text()
                content += '<p>' + content_detail + '</p>'
            print content
            item['content'] = content
            item['date'] = self.date
            item['author'] = author
            item['source'] = self.source
            yield item

        except:
            traceback.print_exc()



