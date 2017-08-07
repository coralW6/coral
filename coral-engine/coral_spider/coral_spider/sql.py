linux_idc_sql = """
    create table coral.articles_detail(
        id int(10) unsigned NOT NULL AUTO_INCREMENT COMMENT '自增长',
        create_time date NOT NULL COMMENT  '创建时间',
        title varchar(120) NOT NULL DEFAULT '' COMMENT '文章标题',
        author varchar(40) NOT NULL DEFAULT '0' COMMENT '文章作者',
        source varchar(80) NOT NULL DEFAULT '' COMMENT '文章来源',
        original_url varchar(120) NOT NULL DEFAULT '' COMMENT '原始链接',
        content text NOT NULL COMMENT '文章内容',
        is_del tinyint(2) DEFAULT '0' COMMENT '是否删除 0:否, 1:是',
        picture_id int(10) DEFAULT '0' COMMENT '文章对应的图片id',
        PRIMARY KEY (id),
        KEY index_create_time (create_time)
    ) ENGINE=InnoDB AUTO_INCREMENT=0 DEFAULT CHARSET=utf8 COMMENT='抓取的文章详情';
"""