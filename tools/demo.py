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

CLASSES = ('__background__',
           'symbol')

NETS = {'vgg16': ('VGG16',
                  'VGG16_faster_rcnn_final.caffemodel'),
        'zf': ('ZF',
                  'ZF_faster_rcnn_final.caffemodel'),
        'zfsymbol' : ('ZF',
                    'ZF_symbol_faster_rcnn_final.caffemodel')}


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

def demo(net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
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
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    prototxt = os.path.join(cfg.MODELS_DIR, NETS[args.demo_net][0],
                            'faster_rcnn_alt_opt', 'faster_rcnn_test.pt')
    caffemodel = os.path.join(cfg.DATA_DIR, 'faster_rcnn_models',
                              NETS[args.demo_net][1])

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

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in xrange(2):
        _, _= im_detect(net, im)

    im_names = ['1431941529574940_25554184_1433761214029.jpg',
		 '1433677652424751_0_1433761643208.jpg',
		 '1441354419481965_2261324_1441807550442.jpg',
		 '1433761314316492_14907420_1433761394453.jpg',
		 '1452156280284124_3026910_1452157195458.jpg',
		 '1435979508510124_1964069_1442705607760.jpg',
		 '1450176977515023_3725293_1450186343640.jpg',
		 '1432851167055358_42874736_1433761633083.jpg',
		 '1438057127324203_3324735_1443003666953.jpg',
		 '1432963890422826_42965309_1433761667243.jpg',
		 '1433758921021655_43591929_1433761171532.jpg',
		 '1443770166161171_2961341_1443772629.jpg',
		 '1433760274014401_41320612_1433761395498.jpg',
		 '1442043930270945_3871688_1442044154843.jpg',
		 '1451658673652387_1595089_1452608004513.jpg',
		 '1433747078188238_0_1433761547942.jpg',
		 '1432948178875237_43178147_1433761394613.jpg',
		 '1433761447259234_43797547_1433761933815.jpg',
		 '1433554462356441_43481178_1433761713278.jpg',
		 '1430316989630890_41137319_1433761232811.jpg',
		 '1441299868823047_4664311_1441802914054.jpg',
		 '1433759815772097_0_1433761284879.jpg',
		 '1442657170874929_5435502_1445462038994.jpg',
		 '1440212329755118_1729723_1442552427235.jpg',
		 '1438592533434831_3031565_1442794051445.jpg',
		 '780821820266_0_1433761576093.jpg',
		 '1432963548218541_42964872_1433761682556.jpg',
		 '1441094483028842_1321698_1447763397952.jpg',
		 '1433575078305028_43529856_1433761069831.jpg',
		 '1432811935873648_42848307_1433761713251.jpg',
		 '1418380153360642_0_1433762207747.jpg',
		 '1433761334727159_0_1433761561941.jpg',
		 '1433760803475362_0_1433761714103.jpg',
		 '1433760396879296_0_1433761689871.jpg',
		 '1452310860106240_2244699_1452311082239.jpg',
		 '1429620891497966_40540054_1433761489772.jpg',
		 '1433761040113320_0_1433761225089.jpg',
		 '1433751721906551_43777711_1433761189541.jpg',
		 '1433409072032213_43327069_1433761582372.jpg',
		 '1418380153360642_0_1433762155185.jpg',
		 '1433761449843731_0_1433761530261.jpg',
		 '1452079495711978_2026541_1452399009776.jpg']


    for im_name in im_names:
        print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'
        print 'Demo for data/demo/{}'.format(im_name)
        demo(net, im_name)

    plt.show()
