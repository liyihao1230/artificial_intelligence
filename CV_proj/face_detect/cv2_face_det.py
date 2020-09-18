#!/usr/bin/env python
# coding: utf-8

import numpy as np
import cv2
from matplotlib import pyplot as plt
from PIL import Image
import time
# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams['figure.figsize'] = (5.0, 4.0)

cap = cv2.VideoCapture(0) # 参数为0时调用本地摄像头；url连接调取网络摄像头；文件地址获取本地视频

# 获取网络摄像头，格式：rtsp://username:pwd@ip/
# cap = cv2.VideoCapture('rtsp://username:pwd@ip/')

face_cascade = cv2.CascadeClassifier('/Users/yihaoli/opt/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('/Users/yihaoli/opt/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml')
body_cascade = cv2.CascadeClassifier('/Users/yihaoli/opt/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_upperbody.xml')
time.sleep(1)

if __name__ == '__main__':
    while cap.isOpened():
        ret,frame = cap.read()#读取每一帧
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray,1.1,3,0,(200,100))
        bodies = body_cascade.detectMultiScale(gray, 1.1,3,0,(200,100))
        for(x,y,w,h) in faces:
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,0),2)
            face_gray = gray[y: y + h, x: x + w]
            face_color = img[y: y + h, x: x + w]
            eyes = eye_cascade.detectMultiScale(face_gray)
            for i,(ex, ey, ew, eh) in enumerate(eyes):
                cv2.rectangle(face_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
        for (x,y,w,h) in bodies:
            img = cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        cv2.imshow('img',img)#显示每一帧
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            break
        elif cv2.waitKey(1) & 0xFF==ord('s'):
            #save picture and exit
            cv2.imwrite('camera.png',img)
            cv2.imwrite('face.png',face_color)
            for i,(ex, ey, ew, eh) in enumerate(eyes):
                # print(ex,ey,ew,eh)
                eye_color = face_color[ey:ey+eh,ex:ex+ew]
                cv2.imwrite('eye_%d.png'%i,eye_color)
            break
    cap.release()
    cv2.destroyAllWindows()


# #1.1读取图片imread；展示图片imshow；导出图片imwrite
# #只是灰度图片
# img=cv2.imread('Myhero.jpg',cv2.IMREAD_GRAYSCALE)
# #彩色图片
# img=cv2.imread('Myhero.jpg',cv2.IMREAD_COLOR)
# #彩色以及带有透明度
# img=cv2.imread('Myhero.jpg',cv2.IMREAD_UNCHANGED)
# print(img)
# #设置窗口可自动调节大小
# cv2.namedWindow('image',cv2.WINDOW_NORMAL)
# cv2.imshow('image',img)



