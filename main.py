from flask import Flask, render_template,request,jsonify,Response
from database import mysql_operate
import threading
from qiniumiao import *
from cvs import *
import cv2
import math
import numpy as np
from scipy.special import expit
import time
import aidlite_gpu

aidlite = aidlite_gpu.aidlite()

def upload_img(bucket_name, file_name, file_path):
    # generate token
    token = q.upload_token(bucket_name, file_name)
    put_file(token, file_name, file_path)

# 获得七牛云服务器上file_name的图片外链
def get_img_url(bucket_url, file_name):
    img_url = 'http://%s/%s' % (bucket_url, file_name)
    return img_url

def resize_pad(img):
    size0 = img.shape
    if size0[0]>=size0[1]:
        h1 = 256
        w1 = 256 * size0[1] // size0[0] #'//'运算符在Python中是地板除（取整除）
        padh = 0
        padw = 256 - w1 #计算宽度填充，即原宽度减去应有宽度
        scale = size0[1] / w1 #通过将原图宽度除以应有宽度，得到宽缩放系数
    else:
        h1 = 256 * size0[0] // size0[1] #'//'运算符在Python中是地板除（取整除）
        w1 = 256
        padh = 256 - h1 #计算高度填充，即原高度减去应有高度
        padw = 0
        scale = size0[0] / h1 #通过将原图高度除以应有高度，得到高缩放系数

    #将填充值细化为1、2,分别代表地板除以2和天花板除以2得到的值
    padh1 = padh//2 #地板除 floor division
    padh2 = padh//2 + padh%2 #天花板除 ceiling division
    padw1 = padw//2
    padw2 = padw//2 + padw%2

    #对原图进行重定义大小操作
    #注意：cv2的resize函数需作用于Numpy.array对象
    img1 = cv2.resize(img, (w1,h1))
    img1 = np.pad(img1, ((padh1, padh2), (padw1, padw2), (0,0)), 'constant', constant_values=(0,0))
    pad = (int(padh1 * scale), int(padw1 * scale))
    #而这里重定义大小为128*128
    img2 = cv2.resize(img1, (128,128))
    return img1, img2, scale, pad
    
'''
图像逆归一化
'''
def denormalize_detections(detections, scale, pad):
    detections[:, 0] = detections[:, 0] * scale * 256 - pad[0]
    detections[:, 1] = detections[:, 1] * scale * 256 - pad[1]
    detections[:, 2] = detections[:, 2] * scale * 256 - pad[0]
    detections[:, 3] = detections[:, 3] * scale * 256 - pad[1]

    detections[:, 4::2] = detections[:, 4::2] * scale * 256 - pad[1]
    detections[:, 5::2] = detections[:, 5::2] * scale * 256 - pad[0]
    return detections

#锚点框函数，得到真实坐标
def _decode_boxes(raw_boxes, anchors):
    """
    通过使用锚点框，将预测结果转换为真实坐标，同一时间处理整个batch。
    """
    #boxes是一个与raw_boxes相同形状的全为0的数组
    boxes = np.zeros_like(raw_boxes)
    x_center = raw_boxes[..., 0] / 128.0 * anchors[:, 2] + anchors[:, 0]
    y_center = raw_boxes[..., 1] / 128.0 * anchors[:, 3] + anchors[:, 1]

    w = raw_boxes[..., 2] / 128.0 * anchors[:, 2]
    h = raw_boxes[..., 3] / 128.0 * anchors[:, 3]

    boxes[..., 0] = y_center - h / 2.  # ymin
    boxes[..., 1] = x_center - w / 2.  # xmin
    boxes[..., 2] = y_center + h / 2.  # ymax
    boxes[..., 3] = x_center + w / 2.  # xmax

    for k in range(4):
        offset = 4 + k*2
        keypoint_x = raw_boxes[..., offset    ] / 128.0 * anchors[:, 2] + anchors[:, 0]
        keypoint_y = raw_boxes[..., offset + 1] / 128.0 * anchors[:, 3] + anchors[:, 1]
        boxes[..., offset    ] = keypoint_x
        boxes[..., offset + 1] = keypoint_y

    return boxes

#将张量转化为正确的探测值
def _tensors_to_detections(raw_box_tensor, raw_score_tensor, anchors):
    #检测盒，存放由预测结果转化成的真实坐标
    detection_boxes = _decode_boxes(raw_box_tensor, anchors)
    
    thresh = 100.0
    raw_score_tensor = np.clip(raw_score_tensor, -thresh, thresh)
    detection_scores = expit(raw_score_tensor)
    
    #提示：由于我们把分数张量的最后一个维度舍弃了，又只对一个class进行处理。
    #于是我们就可以使用遮罩来将置信度过低的盒子过滤掉
    mask = detection_scores >= 0.75

    #因为batch的每张图都有不同数目的检测结果
    #我们使用循环，一次处理一个结果
    boxes = detection_boxes[mask]
    scores = detection_scores[mask]
    scores = scores[..., np.newaxis]
    #numpy.hstack等价于np.concatenate,水平拼接
    return np.hstack((boxes, scores))


#nms算法
def py_cpu_nms(dets, thresh):  
    """Pure Python NMS baseline.（NMS: Non-Maximum Suppression, 非极大值抑制.
    顾名思义,抑制不是极大值的元素，可以理解为局部最大搜索）"""  
    x1 = dets[:, 0]  
    y1 = dets[:, 1]  
    x2 = dets[:, 2]  
    y2 = dets[:, 3]  
    scores = dets[:, 12]  

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  
    #从大到小排列，取index  
    order = scores.argsort()[::-1]  
    #keep为最后保留的边框  
    keep = []  
    while order.size > 0:  
    #order[0]是当前分数最大的窗口，之前没有被过滤掉，肯定是要保留的  
        i = order[0]  
        keep.append(dets[i])  
        #计算窗口i与其他所以窗口的交叠部分的面积，矩阵计算
        xx1 = np.maximum(x1[i], x1[order[1:]])  
        yy1 = np.maximum(y1[i], y1[order[1:]])  
        xx2 = np.minimum(x2[i], x2[order[1:]])  
        yy2 = np.minimum(y2[i], y2[order[1:]])  

        w = np.maximum(0.0, xx2 - xx1 + 1)  
        h = np.maximum(0.0, yy2 - yy1 + 1)  
        inter = w * h  
        #交/并得到iou值  
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  
        #ind为所有与窗口i的iou值小于threshold值的窗口的index，其他窗口此次都被窗口i吸收  
        inds = np.where(ovr <= thresh)[0]  
        #下一次计算前要把窗口i去除，所有i对应的在order里的位置是0，所以剩下的加1  
        order = order[inds + 1]  
    return keep


#将检测结果转化为边界框
def detection2roi(detection):
    """ 将检测结果转化为有方向的边界框
    改写自：mediapipe/modules/face_landmark/face_detection_front_detection_to_roi.pbtxt
    边界框的中心和大小接由中心计算得到。
    旋转角从向量kp1和kp2计算得到，与theta0相关。
    边界框经过scale进行缩放和dy进行平移。
    """
    kp1 = 2
    kp2 = 3
    theta0 = 90 * np.pi / 180
    dscale = 1.5
    dy = 0.
    xc = detection[:,4+2*kp1]
    yc = detection[:,4+2*kp1+1]
    x1 = detection[:,4+2*kp2]
    y1 = detection[:,4+2*kp2+1]
    scale = np.sqrt((xc-x1)**2 + (yc-y1)**2) * 2

    yc += dy * scale
    scale *= dscale

    # compute box rotation
    x0 = detection[:,4+2*kp1]
    y0 = detection[:,4+2*kp1+1]
    x1 = detection[:,4+2*kp2]
    y1 = detection[:,4+2*kp2+1]
    theta = np.arctan2(y0-y1, x0-x1) - theta0
    return xc, yc, scale, theta

#提取关注区域(roi:regions of interest)
def extract_roi(frame, xc, yc, theta, scale):

    # 转化边界框的各点
    points = np.array([[-1, -1, 1, 1],
                        [-1, 1, -1, 1]], dtype=np.float32).reshape(1,2,4)
    points = points * scale.reshape(-1,1,1)/2
    theta = theta.reshape(-1, 1, 1)
    #numpy.concatenate用于连接两个numpy数组
    R = np.concatenate((
        np.concatenate((np.cos(theta), -np.sin(theta)), 2), #numpy.cos、numpy.sin计算弧度为theta的余弦、正弦值
        np.concatenate((np.sin(theta), np.cos(theta)), 2),
        ), 1)
    center = np.concatenate((xc.reshape(-1,1,1), yc.reshape(-1,1,1)), 1)
    points = R @ points + center

    # use the points to compute the affine transform that maps 
    # these points back to the output square
    res = 256
    points1 = np.array([[0, 0, res-1],
                        [0, res-1, 0]], dtype=np.float32).T
    affines = []
    imgs = []
    for i in range(points.shape[0]):
        pts = points[i, :, :3].T
        print('pts', pts.shape, points1.shape, pts.dtype, points1.dtype)
        M = cv2.getAffineTransform(pts, points1)
        img = cv2.warpAffine(frame, M, (res,res))#, borderValue=127.5)
        imgs.append(img)
        affine = cv2.invertAffineTransform(M).astype('float32')
        affines.append(affine)
    if imgs:
        imgs = np.stack(imgs).astype(np.float32) / 255.#/ 127.5 - 1.0
        affines = np.stack(affines)
    else:
        imgs = np.zeros((0, 3, res, res))
        affines = np.zeros((0, 2, 3))

    return imgs, affines, points

#将特征点逆归一化
def denormalize_landmarks(landmarks, affines):
    # landmarks[:,:,:2] *= 256
    for i in range(len(landmarks)):
        landmark, affine = landmarks[i], affines[i]
        landmark = (affine[:,:2] @ landmark[:,:2].T + affine[:,2:]).T
        landmarks[i,:,:2] = landmark
    return landmarks
    
#绘制检测结果
def draw_detections(img, detections, with_keypoints=True):
    if detections.ndim == 1:
        detections = np.expand_dims(detections, axis=0)

    n_keypoints = detections.shape[1] // 2 - 2

    for i in range(detections.shape[0]):
        ymin = detections[i, 0]
        xmin = detections[i, 1]
        ymax = detections[i, 2]
        xmax = detections[i, 3]
        
        start_point = (int(xmin), int(ymin))
        end_point = (int(xmax), int(ymax))
        #cv2.rectangle用于绘制矩形，参数分别为：矩形的图、左上角点、右下角点、矩形颜色、线条的粗细。
        img = cv2.rectangle(img, start_point, end_point, (255, 0, 0), 1) 

        if with_keypoints:
            for k in range(n_keypoints):
                kp_x = int(detections[i, 4 + k*2    ])
                kp_y = int(detections[i, 4 + k*2 + 1])
                cv2.circle(img, (kp_x, kp_y), 2, (0, 0, 255), thickness=2)
    return img

#绘制边界框
def draw_roi(img, roi):
    for i in range(roi.shape[0]):
        (x1,x2,x3,x4), (y1,y2,y3,y4) = roi[i]
        cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,0), 2)
        cv2.line(img, (int(x1), int(y1)), (int(x3), int(y3)), (0,255,0), 2)
        cv2.line(img, (int(x2), int(y2)), (int(x4), int(y4)), (0,0,0), 2)
        cv2.line(img, (int(x3), int(y3)), (int(x4), int(y4)), (0,0,0), 2)

#绘制特征点
def draw_landmarks(img, points, connections=[], color=(255, 255, 0), size=2):
    n = 0
    for point in points:
        x, y = point
        x, y = int(x), int(y)
        #cv2.circle用于画圈，参数分别为：圆的图、圆心、半径、圆的颜色、画圆的线条的粗细（、画圆的线的类型、中心坐标和半径值中的小数位数）。
        cv2.circle(img, (x, y), size, color, thickness=size)
        #cv2.putText(img,str(n),(x,y),cv2.FONT_HERSHEY_COMPLEX,0.4,[0,0,255])
        n = n + 1
    for connection in connections:
        x0, y0 = points[connection[0]]
        x1, y1 = points[connection[1]]
        x0, y0 = int(x0), int(y0)
        x1, y1 = int(x1), int(y1)
        cv2.line(img, (x0, y0), (x1, y1), (255,255,255), size)

def detect(landmarks):
    nose_point=landmarks[0]
    lhand_point=landmarks[15]
    rhand_point=landmarks[16]
    lshoulder_point=landmarks[11]
    rshoulder_point=landmarks[12]
    lbuttock_point=landmarks[23]
    rbuttock_point=landmarks[24]
    # 两个参数，1为纵坐标，0为横坐标
    avg_line=(lshoulder_point[0]+rshoulder_point[0])/2
    avg_buttock=(lbuttock_point[1]+rbuttock_point[1])/2
    avg_hand=(lhand_point[1]+rhand_point[1])/2
    half_body=abs(lshoulder_point[0]-rshoulder_point[0])
    center_bukkock=(lbuttock_point[0]+rbuttock_point[0])/2
    height_shoulder=(lshoulder_point[1]+rshoulder_point[1])/2
    if abs(nose_point[0]-avg_line)>(half_body*0.25):
        return 1
    elif not (lhand_point[0]-rshoulder_point[0]< half_body*1.75 and lhand_point[0]-lshoulder_point[0]<half_body*1.75):
        return 2
    elif not (rhand_point[0]-rshoulder_point[0]< half_body*1.75 and rhand_point[0]-lshoulder_point[0]<half_body*1.75):
        return 2
    elif rhand_point[1]<height_shoulder or lhand_point[1]<height_shoulder:
        return 3
    elif avg_hand>avg_buttock or lhand_point[1]>avg_buttock or rhand_point[1]>avg_buttock:
        return 3
    elif abs(center_bukkock-avg_line)>(half_body*0.5):
        return 4
    else:
        return 5

app = Flask(__name__, static_folder='static', template_folder='templates')

#开始导航界面
@app.route('/')
def index():
    return render_template('index.html')

#考生端界面
@app.route('/students')
def f_students():
    return render_template('student.html')

@app.route('/call_python_function', methods=['POST'])
def call_python_function():
    data = request.get_json()
    # 调用你的Python函数，并传递data
    #fqiniumiao()
    result = my_python_function(data)
    return jsonify(result)

def my_python_function(data):
    #从此开始cv检测模块
    photonum = 1
    #指定模型路径
    #pose_detection用于检测体姿
    #pose_landmark_upper_body用于检测上身
    model_path = 'models/pose_detection.tflite'
    model_pose = 'models/pose_landmark_upper_body.tflite'
    inShape =[1 * 128 * 128 *3*4,]
    outShape = [1*896*12*4, 1*896*1*4]
    print('gpu:',aidlite.ANNModel(model_path,inShape,outShape,4,0))
    aidlite.set_g_index(1)
    inShape =[1 * 256 * 256 *3*4,]
    outShape = [1*155*4, 1*1*4, 1*128*128*1*4]
    print('gpu:',aidlite.ANNModel(model_pose,inShape,outShape,4,0))
    POSE_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,7),
    (0,4), (4,5), (5,6), (6,8),
    (9,10),
    (11,13), (13,15), (15,17), (17,19), (19,15), (15,21),
    (12,14), (14,16), (16,18), (18,20), (20,16), (16,22),
    (11,12), (12,24), (24,23), (23,11)
    ]
    #指定锚框
    anchors = np.load('models/anchors.npy')
    cap=cvs.VideoCapture(-1)
    dict1={1:'HeadMoving!',2:"PassingThings!",3:"HandLeaving!",4:"BodyInclining!",5:"Normal"}
    dict2={1:0,2:0,3:0,4:0,5:0}
    while True:
        image = cvs.read()
        if image is None:
            continue
        image_roi=cv2.flip(image,1)
        frame = cv2.cvtColor(image_roi, cv2.COLOR_BGR2RGB)
        img1, img2, scale, pad = resize_pad(frame)
        img2 = img2.astype(np.float32)
        img2 = img2 / 255.# 127.5 - 1.0
        start_time = time.time()   
        aidlite.set_g_index(0)
        aidlite.setTensor_Fp32(img2,128,128)
        #开始运行 
        aidlite.invoke()
        bboxes  = aidlite.getTensor_Fp32(0).reshape(896, -1)
        scores  = aidlite.getTensor_Fp32(1)
        detections = _tensors_to_detections(bboxes, scores, anchors)
        #进行归一化
        normalized_pose_detections = py_cpu_nms(detections, 0.3)
        normalized_pose_detections  = np.stack(normalized_pose_detections ) if len(normalized_pose_detections ) > 0 else np.zeros((0, 12+1))
        #逆归一化
        pose_detections = denormalize_detections(normalized_pose_detections, scale, pad)
        if len(pose_detections) >0:
            xc, yc, scale, theta = detection2roi(pose_detections)
            img, affine, box = extract_roi(frame, xc, yc, theta, scale)
            aidlite.set_g_index(1)
            aidlite.setTensor_Fp32(img,256,256)
        #开始预测
            aidlite.invoke()
            flags  = aidlite.getTensor_Fp32(1).reshape(-1,1)
            normalized_landmarks = aidlite.getTensor_Fp32(0).copy().reshape(1, 31, -1)
            mask = aidlite.getTensor_Fp32(2)
            landmarks = denormalize_landmarks(normalized_landmarks, affine)
        # print('out', normalized_landmarks.shape, affine.shape, landmarks.shape, flags)
            #draw_roi(image_roi, box)
            #t = (time.time() - start_time)
            #lbs = 'Fps: '+ str(int(100/t)/100.)+" ~~ Time:"+str(t*1000) +"ms"
            #cvs.setLbs(lbs) 
            for i in range(len(flags)):
                landmark, flag = landmarks[i], flags[i]
                if flag>.5:
                    draw_landmarks(image_roi, landmark[:,:2], POSE_CONNECTIONS, size=2)
        #显示状态
            status=int(detect(landmark[:,:2]))
            cv2.putText(image_roi, f'Student Status:{dict1[status]}', (5,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            dict2[status] = dict2[status] + 1
            if dict2[status]>4 and status!=5:
                cv2.imwrite(f'/home/aidlux/photos/image_{dict1[status]}{photonum}.jpg', image_roi)
                imagename = "image_" + dict1[status]+str(photonum)+".jpg"
                photonum = photonum + 1
                dict2[status] = 0

    #显示边界框
        cvs.imshow(image_roi)
    print(data)  # 打印接收到的数据
    return {'message': '您的考试已开始！', 'received_data': data}

#教师端界面
@app.route('/teacher')
def f_teacher():
    return render_template('teacher.html')

@app.route('/call_python_function2', methods=['POST'])
def call_python_function2():
    data = request.get_json()
    # 调用你的Python函数，并传递data
    result = my_python_function2(data)
    return jsonify(result)

def my_python_function2(data):
    # 你的Python函数实现
    print(data)  # 打印接收到的数据
    return {'message': '请在windows系统教师端执行此操作！', 'received_data': data}

#执行该flask应用前，Linux终端需要提前运行：export FLASK_ENV=development,
# /bin/python3 /home/aidlux/exam_web/code/main.py
if __name__ == '__main__':
    app.run(port=8088)

flask_thread=threading.Thread(target=app.run,args=('0.0.0.0','9099'))
flask_thread.daemon=True
flask_thread.start()

access_key = 'GCF8V4wa8tuUgxHmTqPQ7YQBz5aGsVUaxTTcSRJD'
secret_key = '9NxGBz10gnUqvRDUtQFDvAYzlL868areFLkis9O4'
bucket_name = 'photobysc171-2'
bucket_url = 'sg4wcemik.hn-bkt.clouddn.com'
q = Auth(access_key, secret_key)
cdn_manager = CdnManager(q)
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
