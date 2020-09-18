import time
import shutil
import cv2
import math
import json
import numpy as np
from PIL import Image
from glob import glob
import fengzhuang_chinese_ocr
import seg_resize as sg
from matplotlib import pyplot as plt
from flask import Flask,request
app = Flask(__name__)

@app.route('/ocr')
def show_text():
	image = cv2.imread('shhsh_10245398_101_20200425201009_1_622918.png')
	proj,ratio,w = sg.get_proj_n_ratio(image)
	bbox = []
	text = []
	seg_num, seg_list = sg.get_seg_list(proj,ratio,w)
	for i in range(seg_num):
		# out = file_out+str(i)+'.png'
		sub_img = image[seg_list[i]:seg_list[i+1],:w]
		sub_img = sg.resize_image(sub_img)
		result,image_framed = fengzhuang_chinese_ocr.one_img_get_text(sub_img)
	    
	    #show text
		for key in result:
	        #print(result[key][1])
			text.append(result[key][1])
	# print(text)
	return json.dumps(text,ensure_ascii = False)

if __name__ == '__main__':
	app.run(host = '0.0.0.0',port = 50009)