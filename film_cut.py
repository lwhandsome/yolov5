import cv2
import os


vc = cv2.VideoCapture(r'C:\Users\lwhan\Documents\GitHub\yolov5\video1.mp4')  # 读入视频文件，命名cv
n = 1  # 计数
 
if vc.isOpened():  # 判断是否正常打开
    rval, frame = vc.read()
else:
    rval = False
 
timeF = 45  # 视频帧计数间隔频率
 
save_path = r"C:\Users\lwhan\Documents\Github\yolov5\data_practice"  
if not os.path.exists(save_path):
    os.makedirs(save_path)
    print("new folder has been created!")
else:
    print("saving start...")
 
i = 0
while rval:  # 循环读取视频帧
    rval, frame = vc.read()
    
    if (n % timeF == 0):  # 每隔timeF帧进行存储操作
        i += 1
        print(i)
        cv2.imwrite(save_path + r'/{}.jpg'.format(i), frame)
    n = n + 1
    cv2.waitKey(1)
vc.release()
