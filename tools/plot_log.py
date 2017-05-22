#!/usr/bin/env python
# encoding: utf-8

import re
import numpy as np
import matplotlib.pyplot as plt


def str2num(str_list, output_type):
    return [output_type(x) for x in str_list]

def main():
    log_file = "/home/xuxing/workspace-xing/py-faster-rcnn/experiments/logs/faster_rcnn_alt_opt_VGG16_.txt.2017-05-15_22-04-43"
    pattern_itr = re.compile(r"105\]\s+Iteration\s+([\d]+)")
    pattern_rpn = re.compile(r"rpn_cls_loss[\s=]{1,3}([\d\.]+)")
    pattern_box = re.compile(r"rpn_loss_bbox[\s=]{1,3}([\d\.]+)")

    with open(log_file, 'r') as f:
        lines = f.read()
        #itrs = pattern_itr.findall(lines)
        rpns = pattern_rpn.findall(lines)
        boxs = pattern_box.findall(lines)

        itrs = np.array(range(0,80000,20))
        rpns = np.array(str2num(rpns, float))
        rpn1 = rpns[:4000]
        rpn2 = rpns[4000:]
        boxs = np.array(str2num(boxs, float))
        box1 = boxs[:4000]
        box2 = boxs[4000:]
        plt.figure(1)

        plt.sca(plt.subplot(221))
        plt.plot(itrs, rpn1)
        plt.title("RPN Class Loss Stage 1")

	plt.sca(plt.subplot(222))
        plt.plot(itrs, rpn2)
        plt.title("RPN Class Loss Stage 2")

        plt.sca(plt.subplot(223))
        plt.plot(itrs, box1)
        plt.title("RPN Boundary Box Loss Stage 1")

        plt.sca(plt.subplot(224))
        plt.plot(itrs, box2)
        plt.title("RPN Boundary Box Loss Stage 2")

        plt.show()


if "__main__" == __name__:
    main()
