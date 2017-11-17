# -*- coding: utf-8 -*-

import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
home = os.getenv('HOME')
sys.path.append(home + '/personalWorkplace/coral/coral-engine/coral_spider/')

from util.DBFactory import factory

# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: http://doc.scrapy.org/en/latest/topics/item-pipeline.html

db = factory.get_db('coral')

class CoralSpiderPipeline(object):
    def process_item(self, item, spider):
        return item

class MysqlPipeline(object):
    def process_item(self, item, spider):
        date = item.get('date')
        author = item.get('author')
        title = item.get('title')
        source = item.get('source')
        original_url = item.get('original_url')
        content = item.get('content')
        is_del = 0
        picture_id = 0
        print 'MysqlPipeline:', date, author, title, source, content, original_url
        insert_sql = """
            insert into coral.articles_detail (create_time, title, author, source, original_url, content, is_del, picture_id)
            values ("%s", "%s", "%s", "%s", "%s", "%s", %s, %s)
        """ % (date, title, author, source, original_url, content, is_del, picture_id)
        print 'insert_sql:', insert_sql
        db.stat(insert_sql)
        return item

class ChazidianZuowenPipeline(object):
    def process_item(self, item, spider):
        date = item.get('date')
        author = item.get('author')
        title = item.get('title')
        source = item.get('source')
        original_url = item.get('original_url')
        content = item.get('content')
        category = item.get('category')
        grade = item.get('grade')
        is_del = 0
        picture_id = 0
        print 'ChazidianZuowenPipeline:', date, author, title, source, content, original_url
        insert_sql = """
            insert into coral.articles_detail (create_time, title, author, source, original_url, content, is_del, picture_id, category, grade)
            values ("%s", "%s", "%s", "%s", "%s", "%s", %s, %s, "%s", "%s")
        """ % (date, title, author, source, original_url, content, is_del, picture_id, category, grade)
        print 'insert_sql:', insert_sql
        db.stat(insert_sql)
        return item


