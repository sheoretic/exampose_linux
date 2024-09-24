from qiniu import Auth, put_file
from qiniu import CdnManager
import time
import requests
import os
import datetime

# 配置七牛云信息
access_key = 'GCF8V4wa8tuUgxHmTqPQ7YQBz5aGsVUaxTTcSRJD'
secret_key = '9NxGBz10gnUqvRDUtQFDvAYzlL868areFLkis9O4'
bucket_name = 'photobysc171-2'
bucket_url = 'sg4wcemik.hn-bkt.clouddn.com'
q = Auth(access_key, secret_key)
cdn_manager = CdnManager(q)

def upload_img(bucket_name, file_name, file_path):
    # generate token
    token = q.upload_token(bucket_name, file_name)
    put_file(token, file_name, file_path)

# 获得七牛云服务器上file_name的图片外链
def get_img_url(bucket_url, file_name):
    img_url = 'http://%s/%s' % (bucket_url, file_name)
    return img_url

#完成对开发板的实时文件监控，完成对教师的提醒功能
def fqiniumiao():
    folder_path = '/home/aidlux/photos/'
    photonum = 1
    while True:
        file_list = os.listdir(folder_path)
        file_count = len(file_list)
        if file_count != photonum:
            photonum = file_count
            # 需要上传到七牛云上面的图片的路径
            list=os.listdir(folder_path)
            list.sort(key=lambda fn: os.path.getmtime(folder_path+fn) if not os.path.isdir(folder_path+fn) else 0)
            image_up_name = "/home/aidlux/photos/"+str(list[-1])
            # 上传到七牛云后，保存成的图片名称
            timestr = time.strftime("%H%M%S")
            image_qiniu_name = f"detect_image_{timestr}.jpg"
            # 将图片上传到七牛云,并保存成image_qiniu_name的名称
            upload_img(bucket_name, image_qiniu_name, image_up_name)
            # 取出和image_qiniu_name一样名称图片的url
            url_receive = get_img_url(bucket_url, image_qiniu_name)
            print(url_receive)

            # 需要刷新的文件链接,由于不同时间段上传的图片有缓存，因此需要CDN清除缓存，
            urls = [url_receive]
            # URL刷新缓存链接,一天有500次的刷新缓存机会
            refresh_url_result = cdn_manager.refresh_urls(urls)

            # 填写对应的喵码
            id = 'tDWjXTK'
            # 填写喵提醒中，发送的消息，这里放上前面提到的图片外链
            text = "抓拍图片：" + url_receive
            ts = str(time.time())  # 时间戳
            type = 'json'  # 返回内容格式
            request_url = "http://miaotixing.com/trigger?"

            headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.67 Safari/537.36 Edg/87.0.664.47'}

            result = requests.post(request_url + "id=" + id + "&text=" + text + "&ts=" + ts + "&type=" + type,headers=headers)
            print(result)