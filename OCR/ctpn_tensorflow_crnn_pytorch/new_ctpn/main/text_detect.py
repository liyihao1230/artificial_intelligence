# coding=utf-8
import os
import shutil
import sys
import time

import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

#sys.path.append(os.getcwd())
from nets import model_train as model
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector

tf.app.flags.DEFINE_string('test_data_path', 'data/demo/', '')
tf.app.flags.DEFINE_string('output_path', 'data/res/', '')
tf.app.flags.DEFINE_string('gpu', '0', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'new_ctpn/checkpoints_mlt/', '')
tf.app.flags.DEFINE_string('f', '', 'kernel')
FLAGS = tf.app.flags.FLAGS


def get_images():
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

def resize_im(im, scale, max_scale=None):
    f = float(scale) / min(im.shape[0], im.shape[1])
    if max_scale != None and f * max(im.shape[0], im.shape[1]) > max_scale:
        f = float(max_scale) / max(im.shape[0], im.shape[1])
    return cv2.resize(im, None, None, fx=f, fy=f, interpolation=cv2.INTER_LINEAR), f

def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])


def text_detect(image):
    #image, scale = resize_im(image, scale=900, max_scale=1500)
    tf.reset_default_graph()
    with tf.get_default_graph().as_default():
        # 模型参数定义
        input_image = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_image')
        input_im_info = tf.placeholder(tf.float32, shape=[None, 3], name='input_im_info')

        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

        bbox_pred, cls_pred, cls_prob = model.model(input_image)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            # 加载模型
            #ckpt_state = tf.train.get_checkpoint_state(checkpoint_dir)
            #model_path = os.path.join(checkpoint_dir, os.path.basename(ckpt_state.model_checkpoint_path))
            #ckpt_state = tf.train.get_checkpoint_state('/Users/yihaoli/Desktop/text-detection-ctpn-banjin-dev/checkpoints_mlt')
            #model_path = os.path.join('/Users/yihaoli/Desktop/text-detection-ctpn-banjin-dev/checkpoints_mlt', os.path.basename(ckpt_state.model_checkpoint_path))
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            saver.restore(sess, model_path)

            # 预测文本框位置
            img = image
            img, (rh,rw) = resize_image(img)
            image = img
            h, w, c = img.shape
            im_info = np.array([h, w, c]).reshape([1, 3])
            bbox_pred_val, cls_prob_val = sess.run([bbox_pred, cls_prob],
                                                   feed_dict={input_image: [img],
                                                              input_im_info: im_info})

            textsegs, _ = proposal_layer(cls_prob_val, bbox_pred_val, im_info)
            scores = textsegs[:, 0]
            textsegs = textsegs[:, 1:5]

            textdetector = TextDetector(DETECT_MODE='H')
            boxes = textdetector.detect(textsegs, scores[:, np.newaxis], img.shape[:2])
            

            boxes = np.array(boxes, dtype=np.int)

            for i, box in enumerate(boxes):
                    cv2.polylines(img, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),thickness=2)

            img = cv2.resize(img, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
            #cv2.imwrite(os.path.join(FLAGS.output_path, os.path.basename(im_fn)), img[:, :, ::-1])
            #plt.imshow(img)
            #print(boxes)
            boxes = boxes[:,:-1]
            # resize后的recs，复原后标记img，resize后image
    return boxes,img,image