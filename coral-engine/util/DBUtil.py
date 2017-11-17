#!/usr/bin/python
# -*- coding: utf-8 -*-
from collections import defaultdict
import pymysql
pymysql.install_as_MySQLdb()


import MySQLdb
import traceback
import time

class DBUtil():
    def __init__(self, reader, writer):
        if not isinstance(reader, dict):
            raise TypeError
        self.reader = reader
        print 'reader:', reader
        if not writer:
            self.writer = reader
        else:
            self.writer = writer
        # 数据库写连接，使用drds
        self.conn_w = None
        self.conn = None
        self._conn()

    def _conn(self):
        try:
            if not self.conn_w:
                self.conn_w = MySQLdb.connect(**self.writer)
            if not self.conn:
                self.conn = MySQLdb.connect(**self.reader)
            print 'conn'
            return True
        except:
            return False

    def check_conn(self):
        _status = False
        reconn_num = 5
        index = 0
        while not _status and index < reconn_num:
            try:
                self.conn.ping()
                self.conn_w.ping()
                _status = True
            except:
                print '重新连接mysql ', index
                if self._conn():
                    _status = True
                    break
                index += 1
                time.sleep(3)

    def fetch_one(self, sql, params=None, auto_close=True):
        self.check_conn()
        cur = None
        try:
            sql = sql.strip()
            cur = self.conn.cursor()
            if params:
                cur.execute(sql, params)
            else:
                cur.execute(sql)
            result = cur.fetchone()
            return result
        except Exception, e:
            if params:
                print sql % params
            else:
                print sql
            raise e
        finally:
            if cur and auto_close:
                cur.close()

    def fetch_all(self, sql, params=None, auto_close=True):
        """
        返回每行为Tuple
        :param sql:
        :param params:
        :return:
        """
        self.check_conn()
        cur = None
        try:
            sql = sql.strip()
            cur = self.conn.cursor()
            cur.execute(sql, params)
            result = cur.fetchall()
            return result
        except Exception, e:
            if params:
                print sql % params
            else:
                print sql
            raise e
        finally:
            if cur and auto_close:
                cur.close()

    def fetch_data_with_map(self, sql, params=None, auto_close=True):
        """
        返回数据为每行数据为一个map
        :param sql:
        :param params:
        :return:
        """
        cur = None
        try:
            #self.check_conn()
            sql = sql.strip()
            cur = self.conn.cursor(MySQLdb.cursors.DictCursor)
            cur.execute(sql, params)
            result = cur.fetchall()
            return result
        except Exception, e:
            if params:
                print sql % params
            else:
                print sql
            traceback.print_exc()
            raise e
        finally:
            if cur and auto_close:
                cur.close()

    def fetch_one_with_map(self, sql, params=None, auto_close=True):

        """
        返回数据为每行数据为一个map
        :param sql:
        :param params:
        :return:
        """
        self.check_conn()
        cur = None
        try:
            sql = sql.strip()
            cur = self.conn.cursor(MySQLdb.cursors.DictCursor)
            cur.execute(sql, params)
            result = cur.fetchone()
            return result
        except Exception, e:
            if params:
                print sql % params
            else:
                print sql
            raise e
        finally:
            if cur and auto_close:
                cur.close()

    def stat(self, sql, params=None, auto_close=True):
        """
        Insert Delete Update等操作
        :param sql:
        :param params:
        :param auto_close:
        :return: 受影响的行数
        """
        self.check_conn()
        cur = None
        try:
            cur = self.conn_w.cursor()
            infect_rows = cur.execute(sql, params)
            self.conn_w.commit()
            return infect_rows
        except Exception, e:
            if params:
                print sql % params
            else:
                print sql
            raise e
        finally:
            if cur and auto_close:
                cur.close()

    def insert(self, sql, params=None, auto_close=True):
        """
        Insert Delete Update等操作
        :param sql:
        :param params:
        :param auto_close:
        :return: 插入后的主键id
        """
        self.check_conn()
        cur = None
        try:
            cur = self.conn_w.cursor()
            cur.execute(sql, params)
            insert_id = cur.lastrowid
            self.conn_w.commit()
            return insert_id
        except Exception, e:
            if params:
                print sql % params
            else:
                print sql
            raise e
        finally:
            if cur and auto_close:
                cur.close()

    def insert_batch(self, sql, params=None, auto_close=True):
        """
        Insert insert many等操作
        :param sql:
        :param params:
        :param auto_close:
        """
        self.check_conn()
        cur = None
        try:
            cur = self.conn_w.cursor()
            cur.executemany(sql, params)
            self.conn_w.commit()
        except Exception, e:
            if params:
                print sql
		print params
            else:
                print sql
            raise e
        finally:
            if cur and auto_close:
                cur.close()

    def fetch_by_map(self, sql, key_column, value_column=None, params=None, auto_close=True):
        """
        :param sql:
        :param key_column:
        :param value_column:
        :param params:
        :param auto_close:
        :return: {key_column1:value_column1,key_column2:value_column2,...}
        """
        self.check_conn()
        cur = None
        try:
            sql = sql.strip()
            result_map = {}
            # 获取字典cursor,默认cursor取回的是tuple数据
            cur = self.conn.cursor(MySQLdb.cursors.DictCursor)
            cur.execute(sql, params)
            result = cur.fetchall()
            if value_column:
                for row in result:
                    result_map[row[key_column]] = row[value_column]
            else:
                for row in result:
                    result_map[row[key_column]] = row
            return result_map
        except Exception, e:
            if params:
                print sql % params
            else:
                print sql
            raise e
        finally:
            if cur and auto_close:
                cur.close()

    def fetch_by_map_list(self, sql, key_column, value_column=None, params=None, auto_close=True):
        """
        解决key重复的问题，如果相同的key则默认在list上追加
        :param sql:
        :param key_column:
        :param value_column:
        :param params:
        :return:
        """
        self.check_conn()
        cur = None
        try:
            sql = sql.strip()
            result_map = defaultdict(list)
            # 获取字典cursor,默认cursor取回的是tuple数据
            cur = self.conn.cursor(MySQLdb.cursors.DictCursor)
            cur.execute(sql, params)
            result = cur.fetchall()
            if value_column:
                for row in result:
                    result_map[row[key_column]].append(row[value_column])
            else:
                for row in result:
                    result_map[row[key_column]].append(row)
            return result_map
        except Exception, e:
            if params:
                print sql % params
            else:
                print sql
            raise e
        finally:
            if cur and auto_close:
                cur.close()

    def insert_ignore_dict_data(self, table, data_dict, auto_close=True):
        """
        将一个dict对象按照key=column_name, value=column_value插入到数据库中
        :param table:
        :param data_dict:
        :param auto_close:
        :return:
        """
        self.check_conn()
        qmarks = ', '.join(['%s'] * len(data_dict))
        cols = ', '.join(data_dict.keys())
        sql = "INSERT IGNORE INTO %s (%s) VALUES (%s)" % (table, cols, qmarks)
        cur = None
        try:
            cur = self.conn_w.cursor()
            cur.execute(sql, data_dict.values())
            self.conn_w.commit()
        except Exception, e:
            print sql
            raise e
        finally:
            if cur and auto_close:
                cur.close()

    def insert_dict_data(self, table, data_dict, auto_close=True):
        """
        将一个dict对象按照key=column_name, value=column_value插入到数据库中
        :param table:
        :param data_dict:
        :param auto_close:
        :return:
        """
        self.check_conn()
        qmarks = ', '.join(['%s'] * len(data_dict))
        cols = ', '.join(data_dict.keys())
        sql = "INSERT INTO %s (%s) VALUES (%s)" % (table, cols, qmarks)
        cur = None
        try:
            cur = self.conn_w.cursor()
            cur.execute(sql, data_dict.values())
            self.conn_w.commit()
        except Exception, e:
            print sql
            raise e
        finally:
            if cur and auto_close:
                cur.close()

    def close_cursor(self):
        if self.conn and self.conn.cursor:
            self.conn.cursor.close()

    def fetch_many(self, sql, params=None, auto_close=True, show_titles=False):
        """
        返回Generator每次返回Tuple用于大数据查询
        :param sql:
        :param params:
        :rtype: tuple
        """
        self.check_conn()
        batch_size = 10000
        cur = None
        try:
            cur = self.conn.cursor(MySQLdb.cursors.DictCursor)
            cur.execute(sql, params)
            result = cur.fetchmany(batch_size)
            if result and show_titles:
                yield tuple(x[0].split(".")[-1] for x in cur.description)
            while result:
                for r in result:
                    yield r
                result = cur.fetchmany(batch_size)

        except Exception, e:
            if params:
                print sql % params
            else:
                print sql
            import traceback

            traceback.print_exc()
            raise e
        finally:
            if cur and auto_close:
                cur.close()

    def replace_dict_data(self, table, data_dict, auto_close=True):
        """
        将一个dict对象按照key=column_name, value=column_value插入到数据库中
        :param table:
        :param data_dict:
        :param auto_close:
        :return:
        """
        self.check_conn()
        qmarks = ', '.join(['%s'] * len(data_dict))
        cols = ', '.join(data_dict.keys())
        sql = "REPLACE INTO %s (%s) VALUES (%s)" % (table, cols, qmarks)
        cur = None
        try:
            cur = self.conn_w.cursor()
            cur.execute(sql, data_dict.values())
            self.conn_w.commit()
        except Exception, e:
            print sql
            traceback.print_exc()
            raise e
        finally:
            if cur and auto_close:
                cur.close()

    def close_conn(self):
        try:
            sql = "select 1 as active"
            ret_dict = self.fetch_data_with_map(sql)
            for r in ret_dict:
                print 'active:', r['active']
            self.conn.close()
        except:
            print 'active error'
            traceback.print_exc()


if __name__ == '__main__':
    db = DBUtil()
    for i in db.fetch_many("select user_id from cdb.teacher"):
        print i
