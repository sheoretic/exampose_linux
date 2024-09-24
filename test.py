from examtestpose import *
    
#test
def test():
    image = cvs.read()
    #if image is None:
        #continue
    #flip()的作用是使图像进行翻转，cv2.flip(filename, flipcode) 
    #filename：需要操作的图像，flipcode：翻转方式，1水平翻转，0垂直翻转，-1水平垂直翻转
    #如果是前置摄像头，需要翻转图片，想象照镜子的原理
    image_roi=cv2.flip(image,1)

    #cv2.cvtColor(p1,p2) 是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
    #cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式  
    #cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片
    frame = cv2.cvtColor(image_roi, cv2.COLOR_BGR2RGB)
    # frame = np.ascontiguousarray(frame[:,::-1,::-1])
    img1, img2, scale, pad = resize_pad(frame)

    #numpy.astype()是类型转换函数
    img2 = img2.astype(np.float32)
    #将RGB值归一化
    img2 = img2 / 255.# 127.5 - 1.0
    #记录开始时间
    start_time = time.time()   

    #切换到模型0
    aidlite.set_g_index(0)
    #由于fp16的值区间比fp32的值区间小很多，所以在计算过程中很容易出现上溢出（Overflow，>65504 ）和下溢出（Underflow，<6x10^-8  ）的错误，溢出之后就会出现“Nan”的问题
    #分配内存并传入数据
    aidlite.setTensor_Fp32(img2,128,128)
    #开始运行 
    aidlite.invoke()
    bboxes  = aidlite.getTensor_Fp32(0).reshape(896, -1)
    scores  = aidlite.getTensor_Fp32(1)
    
    #将直接张量转化成真实的探测值，输出为一个形状为(b,896,16)的张量。
    detections = _tensors_to_detections(bboxes, scores, anchors)
    #进行归一化
    normalized_pose_detections = py_cpu_nms(detections, 0.3)
    
    #如果归一化后的检测值是一个不为空的列表，则将其堆叠；否则将其设为全零张量
    normalized_pose_detections  = np.stack(normalized_pose_detections ) if len(normalized_pose_detections ) > 0 else np.zeros((0, 12+1))
    #逆归一化
    pose_detections = denormalize_detections(normalized_pose_detections, scale, pad)
    #若所得值不为空
    if len(pose_detections) >0:
        #将检测结果转化为有方向的边界框
        xc, yc, scale, theta = detection2roi(pose_detections)
        #转化边界框的各点
        img, affine, box = extract_roi(frame, xc, yc, theta, scale)
        
        # print(img.shape)
        
        #切换至模型1
        aidlite.set_g_index(1)
        ##由于fp16的值区间比fp32的值区间小很多，所以在计算过程中很容易出现上溢出（Overflow，>65504 ）和下溢出（Underflow，<6x10^-8  ）的错误，溢出之后就会出现“Nan”的问题
        #分配内存并传入数据img
        aidlite.setTensor_Fp32(img,256,256)
        #开始预测
        aidlite.invoke()
        #
        flags  = aidlite.getTensor_Fp32(1).reshape(-1,1)
        normalized_landmarks = aidlite.getTensor_Fp32(0).copy().reshape(1, 31, -1)
        mask = aidlite.getTensor_Fp32(2)
        landmarks = denormalize_landmarks(normalized_landmarks, affine)
        # print('out', normalized_landmarks.shape, affine.shape, landmarks.shape, flags)
        #绘制边界框
        draw_roi(image_roi, box)
        #计算经过时间
        t = (time.time() - start_time)
        # print('elapsed_ms invoke:',t*1000)
        #显示时间
        lbs = 'Fps: '+ str(int(100/t)/100.)+" ~~ Time:"+str(t*1000) +"ms"
        cvs.setLbs(lbs) 
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

'''
入口
'''

photonum = 1
#指定模型路径
#pose_detection用于检测体姿
#pose_landmark_upper_body用于检测上身
model_path = 'models/pose_detection.tflite'
model_pose = 'models/pose_landmark_upper_body.tflite'
# img_path = 'imgs/serena.png'

#检测体姿模型的输入
#图片的尺寸大小，深度学习不需要尺寸过大的图片，128*128就能满足需求
#有rgb3通道
#输入数据数量的单位是字节，1个float是32位也就是4字节，每个数据4个字节
inShape =[1 * 128 * 128 *3*4,]

#4代表4个字节；896指896个框；896*12指明每个框的置信度；这12个数具体意指什么、顺序如何，只有看作者训练时曾怎样定义。若无法找到定义便只能猜测和尝试。
#outShape就是输出数据的数据量 单位是字节,896是值体姿关键点448*2；12代表体姿识别的6个关键点，每个点都有x,y，所以此处为12
#outShape输出图像
outShape = [1*896*12*4, 1*896*1*4]

##4表示4个cpu线程，0表示gpu，-1表示cpu，1表示GPU和CPU共同使用，2表示高通专有加速模式，1表示NNAPI 线程数我在aid上设置的4线程，
# 你可以灵活设置线程数和是否使用gpu+cpu模式；
#NNAPI (神经网络API)是 NDK中的一套API。由C语言实现。NNAPI设计的初衷是扮演底层平台的角色，
# 支撑上层的各种机器语言学习框架(TensorFlow List, Caffe2等)高效的完成推理计算，甚至是构建/训练模型。
# NNAPI有两个显著特点：
#1. 内置于Android系统，从Android8.1系统开始出现在NDK中。
#2. 能够利用硬件加速，使用Android设备的GPU/DSP或各种专门人工智能加速芯片完成神经网络的推理运算，大大提高计算效率。
#NNAPI会根据Android设备的硬件性能，适当的将这些繁重的计算部署到合适的计算单元(CPU & GPU &DSP &神经网络芯片). 从而使用硬件加速功能完成推理运算。
print('gpu:',aidlite.ANNModel(model_path,inShape,outShape,4,0))
aidlite.set_g_index(1)
#图片的尺寸大小，深度学习不需要尺寸过大的图片，这里使用256 * 256
#有rgb3通道
#输入数据数量的单位是字节，1个float是32位也就是4字节，每个数据4个字节
inShape =[1 * 256 * 256 *3*4,]

#4代表4个字，节其他参数具体参照模型
#outShape就是输出数据的数据量 
#outShape输出图像
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


#读取摄像头，0是后置摄像头，1是前置摄像头
cap=cvs.VideoCapture(-1)
dict1={1:'HeadMoving!',2:"PassingThings!",3:"HandLeaving!",4:"BodyInclining!",5:"Normal"}
dict2={1:0,2:0,3:0,4:0,5:0}
while True:
#开始一直循环读取摄像头中的每一帧，直到程序退出
    test()
    '''
    image = cvs.read()
    if image is None:
        continue
    #flip()的作用是使图像进行翻转，cv2.flip(filename, flipcode) 
    #filename：需要操作的图像，flipcode：翻转方式，1水平翻转，0垂直翻转，-1水平垂直翻转
    #如果是前置摄像头，需要翻转图片，想象照镜子的原理
    image_roi=cv2.flip(image,1)

    #cv2.cvtColor(p1,p2) 是颜色空间转换函数，p1是需要转换的图片，p2是转换成何种格式。
    #cv2.COLOR_BGR2RGB 将BGR格式转换成RGB格式  
    #cv2.COLOR_BGR2GRAY 将BGR格式转换成灰度图片
    frame = cv2.cvtColor(image_roi, cv2.COLOR_BGR2RGB)
    # frame = np.ascontiguousarray(frame[:,::-1,::-1])
    img1, img2, scale, pad = resize_pad(frame)

    #numpy.astype()是类型转换函数
    img2 = img2.astype(np.float32)
    #将RGB值归一化
    img2 = img2 / 255.# 127.5 - 1.0
    #记录开始时间
    start_time = time.time()   

    #切换到模型0
    aidlite.set_g_index(0)
    #由于fp16的值区间比fp32的值区间小很多，所以在计算过程中很容易出现上溢出（Overflow，>65504 ）和下溢出（Underflow，<6x10^-8  ）的错误，溢出之后就会出现“Nan”的问题
    #分配内存并传入数据
    aidlite.setTensor_Fp32(img2,128,128)
    #开始运行 
    aidlite.invoke()
    bboxes  = aidlite.getTensor_Fp32(0).reshape(896, -1)
    scores  = aidlite.getTensor_Fp32(1)
    
    #将直接张量转化成真实的探测值，输出为一个形状为(b,896,16)的张量。
    detections = _tensors_to_detections(bboxes, scores, anchors)
    #进行归一化
    normalized_pose_detections = py_cpu_nms(detections, 0.3)
    
    #如果归一化后的检测值是一个不为空的列表，则将其堆叠；否则将其设为全零张量
    normalized_pose_detections  = np.stack(normalized_pose_detections ) if len(normalized_pose_detections ) > 0 else np.zeros((0, 12+1))
    #逆归一化
    pose_detections = denormalize_detections(normalized_pose_detections, scale, pad)
    #若所得值不为空
    if len(pose_detections) >0:
        #将检测结果转化为有方向的边界框
        xc, yc, scale, theta = detection2roi(pose_detections)
        #转化边界框的各点
        img, affine, box = extract_roi(frame, xc, yc, theta, scale)
        
        # print(img.shape)
        
        #切换至模型1
        aidlite.set_g_index(1)
        ##由于fp16的值区间比fp32的值区间小很多，所以在计算过程中很容易出现上溢出（Overflow，>65504 ）和下溢出（Underflow，<6x10^-8  ）的错误，溢出之后就会出现“Nan”的问题
        #分配内存并传入数据img
        aidlite.setTensor_Fp32(img,256,256)
        #开始预测
        aidlite.invoke()
        #
        flags  = aidlite.getTensor_Fp32(1).reshape(-1,1)
        
        normalized_landmarks = aidlite.getTensor_Fp32(0).copy().reshape(1, 31, -1)
        mask = aidlite.getTensor_Fp32(2)
        
        
        landmarks = denormalize_landmarks(normalized_landmarks, affine)
        # print('out', normalized_landmarks.shape, affine.shape, landmarks.shape, flags)
        #绘制边界框
        draw_roi(image_roi, box)
        #计算经过时间
        t = (time.time() - start_time)
        # print('elapsed_ms invoke:',t*1000)
        #显示时间
        lbs = 'Fps: '+ str(int(100/t)/100.)+" ~~ Time:"+str(t*1000) +"ms"
        cvs.setLbs(lbs) 
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
    '''