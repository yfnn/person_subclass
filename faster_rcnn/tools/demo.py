#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import _init_paths
from model.config import cfg, cfg_from_file
from model.test import im_detect, test_net
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse
import pdb

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__',
           'person1', 'person2', 'person3', 'person4')

#NETS = {'vgg16': ('vgg16_faster_rcnn_iter_30000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
#DATASETS= {'eelab': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}
#DATASETS= {'kaist': ('kaist_train',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}
#NETS = {'vgg16': ('vgg16_faster_rcnn_iter_21285.ckpt',),'res101': ('res101_faster_rcnn_iter_4500.ckpt',)}
NUM_CLASSES = 5
DATASETS= {'eelab': ('eelab_train',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}
TESTSETS= ['SET-05', 'SET-06', 'SET-07', 'SET-08']
METHODS = {'conventional': 'vgg16_faster_rcnn_iter_1000.ckpt', 'DF-RCNN-C4': 'vgg16_faster_rcnn_iter_8000.ckpt', 'DF-RCNN-C5': 'vgg16_faster_rcnn_iter_5000.ckpt', 'Dual-YOLO':'vgg16_faster_rcnn_iter_9000.ckpt'}
def vis_detections(im, class_name, dets, image_name, fig, ax, testset, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    #fig, ax = plt.subplots(figsize=(12, 12))
    #ax.imshow(im, aspect='equal')
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

    ax.set_title(('{}   {} detections with '
                  'p({} | box) >= {:.1f}').format(image_name, class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    if not os.path.exists(os.path.join(cfg.DATA_DIR, 'test_result',testset)):
        os.mkdir(os.path.join(cfg.DATA_DIR, 'test_result',testset))
    plt.savefig(os.path.join(cfg.DATA_DIR, 'test_result',testset,image_name))

def demo(sess, net, image_name, testset):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', testset, image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.5
    NMS_THRESH = 0.3
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')

    for cls_ind, cls in enumerate(CLASSES[1:]):
        #if not(cls=='person' or cls=='motorbike' or cls=='bicycle'):
        #    continue
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        vis_detections(im, cls, dets, image_name, fig, ax, testset, thresh=CONF_THRESH)

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    #parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
    #                    choices=NETS.keys(), default='res101')
    parser.add_argument('--methods', dest='methods', help='Which detection method want to use', choices=METHODS.keys(), default='DF-RCNN-C5')
    parser.add_argument('--testset',dest='testset',help='Differrent demo testsets',choices=TESTSETS,default='all')
    #parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
    #                    choices=DATASETS.keys(), default='pascal_voc_0712')
    parser.add_argument('--cfg', dest='cfg_file',help='optional config file', default='/home/yangfan/person_subclass_recognition_system/faster_rcnn/experiments/cfgs/vgg16.yml', type=str)
    parser.add_argument('--comp', dest='comp_mode', help='competition mode',
                        action='store_true')
    parser.add_argument('--map_or_not', dest='map', help='calculate map or mot',default='true', type=str)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    #pdb.set_trace()
    with tf.device('/gpu:0'):
        cfg.TEST.HAS_RPN = True  # Use RPN for proposalsi
        args = parse_args()
        testset = args.testset
        map_or_not = args.map
        demonet = 'vgg16'
        dataset = 'eelab'
        method = args.methods
        cfg_from_file(args.cfg_file)
        filename = 'default/' + os.path.splitext(METHODS[method])[0]

        base_path = '/home/yangfan/person_subclass_recognition_system/faster_rcnn/data/demo/'+testset
        file1 = open('/home/yangfan/person_subclass_recognition_system/faster_rcnn/data/demo/test.txt', 'w')

        images = os.listdir(base_path)
        for img in images:
            #if img[0]=='C':
            file1.write(img[:-4]+'\n')
        file1.close()

        import shutil
        from datasets.factory import get_imdb
        shutil.move('/home/yangfan/person_subclass_recognition_system/faster_rcnn/data/demo/test.txt', '/home/yangfan/person_subclass_recognition_system/faster_rcnn/data/EELABdevkit/data/ImageSets/Main/test.txt')

        imdb = get_imdb('eelab_test')
        imdb.competition_mode(args.comp_mode)

        testset_path=os.path.join('/home/yangfan/person_subclass_recognition_system/faster_rcnn/data/demo',testset)
        num_testset=int(len(os.listdir(testset_path)))

        # model path
        tfmodel = os.path.join('/home/yangfan/person_subclass_recognition_system/faster_rcnn/output', demonet, DATASETS[dataset][0], 'default', METHODS[method])

        if not os.path.isfile(tfmodel + '.meta'):
            raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

        # set config
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
    #tfconfig = tf.ConfigProto(device_count={'GPU':0})
    tfconfig.gpu_options.allow_growth=True


    # init session
    sess = tf.Session(config=tfconfig)
    #pdb.set_trace()
    # load network
    #if demonet == 'vgg16':
    net = vgg16(batch_size=1)
    #elif demonet == 'res101':
    #    net = resnetv1(batch_size=1, num_layers=101)
    #else:
    #    raise NotImplementedError
    net.create_architecture(sess, "TEST", NUM_CLASSES,
                          tag='default', anchor_scales=[8, 16, 32], anchor_ratios=[0.5,1,2])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    print('\n')
    print('--------------------------------------')
    print('Evaluating image set {},{} pairs of images altogether'.format(testset,str(num_testset)))
    #pdb.set_trace()
    time.sleep(3)
    im_names = os.listdir(cfg.DATA_DIR + '/demo/'+testset)#['left000700.png']
    #im_names = ['left000700.png']
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name,testset)
        #pdb.set_trace()
    if map_or_not == 'true':
      conf_threshs = [0.5]
      test_file = open('test_result.txt','w')
      for conf_thresh in conf_threshs:
        test_net(sess, net, imdb, filename, test_file, testset, method, max_per_image=100, thresh=conf_thresh)
        test_file.write('\n')
      test_file.close()
      if os.path.exists('/home/yangfan/person_subclass_recognition_system/faster_rcnn/data/EELABdevkit/annotations_cache/imagesetfile_annots.pkl'):
        os.remove('/home/yangfan/person_subclass_recognition_system/faster_rcnn/data/EELABdevkit/annotations_cache/imagesetfile_annots.pkl')

    plt.show()
