import cv2
import dlib
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import os

sys.path.append(os.pardir)
par_dir = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")

cap = cv2.VideoCapture(0) # 参数为0时调用本地摄像头；url连接调取网络摄像头；文件地址获取本地视频

predictor_path = par_dir+'/dlib_model/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)
time.sleep(1)

if __name__ == '__main__':
    while cap.isOpened():
        ret,frame = cap.read()#读取每一帧
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = frame
        faces = detector(img,0)
        if not len(faces):
        	print('Not found any face')
        	break
        for i in range(len(faces)):
        	landmarks = np.matrix([[p.x, p.y] for p in predictor(img, faces[i]).parts()])
        	for point in landmarks:
	            pos = (point[0, 0], point[0, 1])
	            cv2.circle(img, pos, 3, color=(0, 255, 0),thickness=3)
        cv2.imshow('img',img)#显示每一帧
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break
        elif cv2.waitKey(1) & 0xFF==ord('s'):
            #save picture and exit
            cv2.imwrite('camera.png',img)
            break
    cap.release()
    cv2.destroyAllWindows()



# opencv读取图片是BRG通道的，需要专成RGB
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.figure(figsize=(10, 8))
# plt.subplot(121)
# plt.imshow(plt.imread(test_img))
# plt.axis('off')
# plt.subplot(122)
# plt.imshow(img)
# plt.axis('off')
# plt.show()