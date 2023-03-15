import cv2
import time
import os

#数据路径
dir_path = f"./datasets"
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

data_i = 0
dir_path = f"./datasets/date{data_i}"

#遍历图片，检查路径
while os.path.exists(dir_path):
    data_i = data_i + 1
    dir_path = f"./datasets/date{data_i}"

os.makedirs(dir_path)

cap = cv2.VideoCapture(1)

pic_i = 0

try:
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            cv2.imshow("monitor", frame)
            cv2.waitKey(1)
            cv2.imwrite(f"./datasets/date{data_i}/{pic_i}.jpg", frame)
        else:
            break
        
        pic_i = pic_i + 1
        time.sleep(0.1) #读取间隔时间
    
except KeyboardInterrupt:
    pass

cap.release()
cv2.destroyAllWindows()