#!/usr/bin/env python

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from PIL import Image
import math
import os


CLASSES = ('__background__',
           'symbol')

NETS = {'vgg16': ('VGG16',
                  'VGG16_symbol_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'zfsymbol' : ('ZF',
                    'ZF_symbol_faster_rcnn_final.caffemodel')}

output_img_dir = './tools/predict_res'

#inclass_imgs_path = '/home/xuxing/workspace-xing/py-faster-rcnn/data/demo'

#inclass_imgs_path = '/mnt/data/xuxing/symbol-py-faster-rcnn/image3000'
inclass_imgs_path = '/mnt/data/xuxing/symbol-py-faster-rcnn/shijuantu'
#inclass_imgs_path = '/home/xuxing/workspace-xing/blur-proj/oss_blur_rgb/case0-0.5'

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def in_another_rect(dets_bbox):
    """
    rect1[0] #left-top-x
    rect1[1] #left-top-y
    rect1[2] #right-bottom-x
    rect1[3] #right-bottom-y

    rect2[0] #left-top-x
    rect2[1] #left-top-y
    rect2[2] #right-bottom-x
    rect2[3] #right-bottom-y
    """
    inds = []
    for ni,i in enumerate(dets_bbox):
        j_in_i = False
        i_in_j = False
        for nj,j in enumerate(dets_bbox):
            j_in_i = (i[0] < j[0]) and (i[1] < j[1]) and (i[2] > j[2]) and (i[3] > j[3])
            i_in_j = (j[0] < i[0]) and (j[1] < i[1]) and (j[2] > i[2]) and (j[3] > i[3])
        if j_in_i:
            inds.append(ni)
        elif i_in_j:
            inds.append(nj)
        else:
            inds.append(ni)
    return np.array(inds)


def vis_detections_cv(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    # print 'thres %f' % thresh
    inds = np.where(dets[:, -1] >= thresh)[0]
    print "!!!!!!!!!!!!!!"
    print inds
    if len(inds) == 0:
        print "inds = 0, return!"
        return

    inds = in_another_rect(dets[inds, :4])
    print "!!!!!!!!!!!!!!"
    print inds

    # im = im[:, :, (2, 1, 0)].copy() # BGR -> RGB
    # fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        # ax.add_patch(
        #     plt.Rectangle((bbox[0], bbox[1]),
        #                   bbox[2] - bbox[0],
        #                   bbox[3] - bbox[1], fill=False,
        #                   edgecolor='red', linewidth=3.5)
        #     )

        # draw rectangle for each patch
        cv2.rectangle(im, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0,0,255), 2)

        # draw text
        # ax.text(bbox[0], bbox[1] - 2,
        #        '{:s} {:.3f}'.format(class_name, score),
        #        bbox=dict(facecolor='blue', alpha=0.5),
        #        fontsize=14, color='white')
        # cv2.putText(im, '{:s} {:.3f}'.format(class_name, score), (int(bbox[0]), int(bbox[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2, 8)
        # cv2.putText(im, '{:s}'.format(class_name), (int(bbox[0]), int(bbox[1] - 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 1, 8)

    # cv2.imwrite('./tools/test.jpg', im)
    # print 'write ./tools/test.jpg'
    return im

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im_file = os.path.join(inclass_imgs_path, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]): # exclude the nonface class
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        res = vis_detections_cv(im, cls, dets, thresh=CONF_THRESH)

	if res is not None: # only have detection results, then overlpa the im
	    im = res
        else:
	    continue
        # write to image
    if im is not None:
        rect_image_name = ('rect_%s' % image_name)
        cv2.imwrite(os.path.join(output_img_dir, rect_image_name), im)
        print 'save predicted image in %s' % os.path.join(output_img_dir, rect_image_name)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [ZF]',
                        choices=NETS.keys(), default='zf')

    args = parser.parse_args()

    return args

def get_imgs_dir(img_path):
    # get the image list from the img_path
    # img_path = './inclass_images'
    inclass_imgs = []
    for item in os.listdir(inclass_imgs_path):
        if item.endswith('.jpg') or item.endswith('.JPG'):
            # print item
            # item_fullpath = os.path.join(inclass_imgs_path, item)
            inclass_imgs.append(item)
    return inclass_imgs


def resize_canvas(old_image_path, new_image_path,
                  canvas_width=500, canvas_height=500):
    """
    Place one image on another image.

    Resize the canvas of old_image_path and store the new image in
    new_image_path. Center the image on the new canvas.
    """
    im = Image.open(old_image_path)
    old_width, old_height = im.size
    if old_width == old_height:
        return
    canvas_width = max(old_width, old_height)
    canvas_height = max(old_width, old_height)

    # Center the image
    x1 = int(math.floor((canvas_width - old_width) / 2))
    y1 = int(math.floor((canvas_height - old_height) / 2))

    mode = im.mode
    if len(mode) == 1:  # L, 1
        new_background = (255)
    if len(mode) == 3:  # RGB
        new_background = (255, 255, 255)
    if len(mode) == 4:  # RGBA, CMYK
        new_background = (255, 255, 255, 255)

    newImage = Image.new(mode, (canvas_width, canvas_height), new_background)
    newImage.paste(im, (x1, y1, x1 + old_width, y1 + old_height))
    newImage.save(new_image_path)
    im.close()


def add_padding(i):
    files = os.listdir(i)
    for j in files:
        if j.endswith(".jpg") or j.endswith(".png"):
            inpath = os.path.join(i, j)
            outpath = os.path.join(i, j)
            resize_canvas(old_image_path=inpath, new_image_path=outpath)
    print '%s finish add padding.' % i


if __name__ == '__main__':
    # Load caffe model
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

    # caffemodel = './output/faster_rcnn_alt_opt/voc_2007_trainval/ZF_faster_rcnn_final.caffemodel'
    if not os.path.isfile(caffemodel):
        raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(caffemodel))

    if args.cpu_mode:
        caffe.set_mode_cpu()
    else:
        caffe.set_mode_gpu()
        caffe.set_device(args.gpu_id)
        cfg.GPU_ID = args.gpu_id
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)

    print '\n\nLoaded network {:s}'.format(caffemodel)


    # Add padding
    add_padding(inclass_imgs_path)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    # im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
    #             '001763.jpg', '004545.jpg']
    # im_names = ['yingyangwucan_0-020_230.jpg']

    # inclass_imgs_path = '/home/yukinaga/workspace/py-faster-rcnn/data/inclass/lchenglu/img'
    im_names = get_imgs_dir(inclass_imgs_path); # local path

    if im_names is None:
   	print 'no images found in %s' % inclass_imgs_path
    	exit()

    # make output directory
    if not os.path.exists(output_img_dir):
        os.mkdir(output_img_dir) 

    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Detection for image {}'.format(im_name)
        demo(net, im_name)

    plt.show()
