'''
使用cv2检测人脸和双眼, 用dlib检测人脸关键点
'''

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import time
import dlib
import copy


import sys
import os

sys.path.append(os.pardir)
par_dir = os.path.abspath(os.path.dirname(os.getcwd())+os.path.sep+".")

cap = cv2.VideoCapture(0) # 参数为0时调用本地摄像头；url连接调取网络摄像头；文件地址获取本地视频

# 获取网络摄像头，格式：rtsp://username:pwd@ip/
# cap = cv2.VideoCapture('rtsp://username:pwd@ip/')

# predictor_path = './dlib_model/shape_predictor_5_face_landmarks.dat'
predictor_path = par_dir+'/dlib_model/shape_predictor_68_face_landmarks.dat'
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

face_cascade = cv2.CascadeClassifier('/Users/yihaoli/opt/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/yihaoli/opt/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml')
body_cascade = cv2.CascadeClassifier('/Users/yihaoli/opt/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_upperbody.xml')


def dlib_det(img):
    faces = detector(img,0)
    if not len(faces):
        print('Not found any face')
        return
    for i in range(len(faces)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(img, faces[i]).parts()])
        for point in landmarks:
            pos = (point[0, 0], point[0, 1])
            cv2.circle(img, pos, 3, color=(0, 255, 0),thickness=3)
    cv2.imwrite('dlib_det.png',img)


def cv2_det(gray,img):
    faces = face_cascade.detectMultiScale(gray,1.1,3,0,(200,100))
    for(x,y,w,h) in faces:
        img = cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,0),2)
    eyes = eye_cascade.detectMultiScale(gray)
    for i,(ex, ey, ew, eh) in enumerate(eyes):
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 2)
    cv2.imwrite('cv2_det.png',img)


if __name__ == '__main__':
    while cap.isOpened():
        ret,frame = cap.read()#读取每一帧
        img = copy.deepcopy(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv_img = copy.deepcopy(frame)
        dlib_img = copy.deepcopy(frame)
        cv2_det(gray,img)
       	dlib_det(img)
       	cv2.imshow('img',img)#显示每一帧
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print('Quit')
            break
        if cv2.waitKey(1) & 0xFF == ord('c'):
       	    cv2_det(gray,cv_img)
       	    print('Save cv2_img')
       	if cv2.waitKey(1) & 0xFF == ord('d'):
       	    dlib_det(dlib_img)
       	    print('Save dlib_img')
        if cv2.waitKey(1) & 0xFF == ord('s'):
            # save picture and exit
            cv2.imwrite('res_cam.png',img)
            cv2.imwrite('ori_cam.png',frame)
            print('Save res and ori')
            break
    cap.release()
    cv2.destroyAllWindows()



