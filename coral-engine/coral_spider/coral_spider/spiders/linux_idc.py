# -*- coding: utf-8 -*-
# brief 抓取linux公社的文章
# author wanglei
# date 2017-07-14

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
from coral_spider.items import LinuxIdcItem


class linux_idc(Spider):
    name = 'linux_idc'
    custom_settings = {
        'ITEM_PIPELINES': {'coral_spider.pipelines.MysqlPipeline': 5,}
    }
    def __init__(self):
        try:
            self.redis_key = 'linux_idc_url'
            self.origin_url = "http://www.linuxidc.com/"
            self.start_url = "http://www.linuxidc.com/it/"
            self.log = Logger('linux_idc.log')
            self.redis = Redis()
            self.date = datetime.now().strftime('%Y-%m-%d')
            self.author = 'Linux'
            self.source = 'linux公社'
            self.max_count = 10
        except:
            traceback.print_exc()

    def start_requests(self):
        try:
            self.log.info('开始抓取文章 %s' % self.date)
            meta = {}
            meta['max_count'] = self.max_count
            yield Request(self.start_url, callback=self.parse_urls, dont_filter=True, meta=meta)
        except:
            traceback.print_exc()

    def parse_urls(self, response):
        try:
            count = 0
            r = response
            meta = r.meta
            html = r.body_as_unicode()
            html_pq = pq(html)
            article_urls = html_pq('div.title > a')
            for a in article_urls:
                url = pq(a).attr('href').replace('../', '')
                url = self.origin_url + url
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
                meta['url'] = url
                yield Request(url, self.parse_detail, meta=meta, dont_filter=True)
            self.log.info('抓取完成,count=%s' % (count-1))
            # next_pages = html_pq('div.pager>ul>li>a')
            # if next_pages:
            #     for p in next_pages:
            #         page = pq(p)
            #         if '»' in page.text():
            #             next_url = self.origin_url + page.attr('href').replace('../', '')
            #             print next_url
            #             yield Request(next_url, self.parse_urls, dont_filter=True)

        except:
            traceback.print_exc()

    def parse_detail(self, response):
        try:
            item = LinuxIdcItem()
            r = response
            html = r.body_as_unicode()
            html_pq = pq(html)
            meta = r.meta
            title = html_pq('h1.aTitle').text()
            item['title'] = title
            item['original_url'] = meta['url']
            content = ''
            content_p_list = html_pq('#content > p')
            for p in content_p_list:
                content_detail = pq(p).text()
                if '更新链接地址' in content_detail:
                    continue
                content += '<p>' + content_detail + '</p>'
            item['content'] = content
            item['date'] = self.date
            item['author'] = self.author
            item['source'] = self.source
            yield item

        except:
            traceback.print_exc()



