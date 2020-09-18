#-*- coding:utf-8 -*-
import os
import ocr
import time
import shutil
import numpy as np
from PIL import Image
from glob import glob


def get_text(image_files): 
    result_dir = './test_result'
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.mkdir(result_dir)
    results = []
    images = []
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
        results.append(result)
        images.append(image_framed)
    return results, images

def one_img_get_text(image):
    
    result, image_framed = ocr.model(image)
    return result,image_framed