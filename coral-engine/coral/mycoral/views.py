from django.shortcuts import render
from django.http import HttpResponse
# Create your views here.

import MySQLdb

def test(request):
    print 'test111'
    conn = MySQLdb.Connect(host="127.0.0.1",user="wanglei",passwd="111111",db="coral",charset="utf8")
    cursor = conn.cursor()
    ret = cursor.execute('select 1')
    print ret, 1

    print 111, conn
    return HttpResponse('Hello World')
