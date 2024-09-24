# -*- coding: utf-8 -*-
from qiniu import Auth
from qiniu import BucketManager
import requests
import os

#配置七牛云信息
access_key = 'GCF8V4wa8tuUgxHmTqPQ7YQBz5aGsVUaxTTcSRJD'
secret_key = '9NxGBz10gnUqvRDUtQFDvAYzlL868areFLkis9O4'

q = Auth(access_key, secret_key)
bucket = BucketManager(q)

bucket_name = 'photobysc171-2'
# 前缀
prefix = None
# 列举条目，即需要的数据条数，1-1000，默认200
limit = 200
# 列举出除'/'的所有文件以及以'/'为分隔的所有前缀
delimiter = None
# 标记
marker = None
#保存到本地的地址
path = 'E:/test/'

ret, eof, info = bucket.list(bucket_name, prefix, marker, limit, delimiter)
for i in ret['items']:
    print(i['key'])
    base_url = 'http://sg4wcemik.hn-bkt.clouddn.com/'+i['key']
    print(base_url)

    #如果空间有时间戳防盗链或是私有空间，可以调用该方法生成私有链接
    private_url = q.private_download_url(base_url, expires=100)
    print(private_url)

    r = requests.get(private_url)

    if r.content:
        if not os.path.exists(path):
            os.makedirs(path)
        file = open(path + i['key'], "wb")
        file.write(r.content)
        file.flush()
        file.close()
