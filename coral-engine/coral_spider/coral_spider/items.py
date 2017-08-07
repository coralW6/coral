# -*- coding: utf-8 -*-

# Define here the models for your scraped items
#
# See documentation in:
# http://doc.scrapy.org/en/latest/topics/items.html

import scrapy
from scrapy.item import Item, Field
from datetime import datetime

class CoralSpiderItem(scrapy.Item):
    # define the fields for your item here like:
    # name = scrapy.Field()
    pass

class LinuxIdcItem(Item):
    date = Field()
    author = Field()
    title = Field()
    source = Field()
    content = Field()
    original_url = Field()

class ChazidianZuowenItem(Item):
    date = Field()
    author = Field()
    title = Field()
    source = Field()
    content = Field()
    original_url = Field()
    category = Field()
    grade = Field()


