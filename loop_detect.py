import threading
import cv2
from detector import Detector
import time
# import network

class Cap:
    def __init__(self, cam_id=0):
        #摄像机初始化
        self.camera = cv2.VideoCapture(cam_id)
        ret, self.img = self.camera.read()
        print("INIT")
        if not ret:
            print("READ IMG ERROR")
        self.mtx = threading.Lock()
    
    def stop(self):
        self.stop_flag = True
        self.thread_run.join()
    
    def start(self):
        self.stop_flag = False
        self.thread_run = threading.Thread(target=self.__run)
        self.thread_run.start()
    
    def __run(self):
        while not self.stop_flag:
            self.mtx.acquire()
            ret, self.img = self.camera.read()
            self.mtx.release()
            if ret:
                time.sleep(0.003)
            else:
                print("CAM_ERROR")

    def read(self):
        self.mtx.acquire()
        img0 = self.img.copy()
        self.mtx.release()
        
        return img0
    
if __name__ == '__main__':
    cam = Cap(cam_id=1)  #摄像设备
    cam.start()
    detector = Detector(classes=[0,1,2], conf_thres=0.6, view_img=True)

    while True:
        st = time.time()
        
        img = cam.read()
        
        det, im0 = detector.run(img)
        
        cv2.imshow("monitor", im0)
        if cv2.waitKey(1) == 27:
            cam.stop()
            break
        
        cx = -1
        cy = -1
        for *xyxy, conf, cls in reversed(det):
            cx = (xyxy[0] + xyxy[2]) / 2
            cy = (xyxy[1] + xyxy[3]) / 2
            
            px = -(cy / 240 - 1)
            py = -(cx / 320 - 1)
            
            print(f"classes: {cls}, conf: {conf}, center: ({cx}, {cy}) -> ({px}, {py})")

        ed = time.time()
    
    cv2.destroyAllWindows()