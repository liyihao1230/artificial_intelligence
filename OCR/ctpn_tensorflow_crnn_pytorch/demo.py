#-*- coding:utf-8 -*-
import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'
import ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob
import cv2
image_files = glob('./test_images/*.*')


if __name__ == '__main__':
    result_dir = './test_result'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)

    for image_file in sorted(image_files):
        image = np.array(Image.open(image_file).convert('RGB'))
        t = time.time()
        result, image_framed = ocr.model(image)
        output_file = os.path.join(result_dir, image_file.split('/')[-1])
        Image.fromarray(image_framed).save(output_file)
        print("Mission complete, it took {:.3f}s".format(time.time() - t))
        print("\nRecognition Result:\n")
        for key in result:
            print(result[key][1])

        #存放识别文件txt
        output_txt = os.path.join(result_dir, os.path.splitext(os.path.basename(output_file))[0])
        output_txt = output_txt+'.txt'
        with open(output_txt,'w') as f:
            for key in result:
                #print(result[key][1])
                line = result[key][1]+'\r\n'
                f.writelines(line)

