# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Zheqi he, Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.test import test_net
from model.config import cfg, cfg_from_file, cfg_from_list
from datasets.factory import get_imdb
import argparse
import pprint
import time, os, sys

import tensorflow as tf
import pdb
from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1
from nets.mobilenet_v1 import mobilenetv1

def parse_args():
  """
  Parse input arguments
  """
  parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
  parser.add_argument('--cfg', dest='cfg_file',
            help='optional config file', default=None, type=str)
  parser.add_argument('--model', dest='model',
            help='model to test',
            default=None, type=str)
  parser.add_argument('--imdb', dest='imdb_name',
            help='dataset to test',
            default='voc_2007_test', type=str)
  parser.add_argument('--comp', dest='comp_mode', help='competition mode',
            action='store_true')
  parser.add_argument('--num_dets', dest='max_per_image',
            help='max number of detections per image',
            default=100, type=int)
  parser.add_argument('--tag', dest='tag',
                        help='tag of the model',
                        default='', type=str)
  parser.add_argument('--net', dest='net',
                      help='vgg16, res50, res101, res152',
                      default='res50', type=str)
  parser.add_argument('--set', dest='set_cfgs',
                        help='set config keys', default=None,
                        nargs=argparse.REMAINDER)

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()
  return args

if __name__ == '__main__':
  args = parse_args()
  pdb.set_trace()

  print('Called with args:')
  print(args)

  if args.cfg_file is not None:
    cfg_from_file(args.cfg_file)
  if args.set_cfgs is not None:
    cfg_from_list(args.set_cfgs)

  print('Using config:')
  pprint.pprint(cfg)

  # if has model, get the name from it
  # if does not, then just use the inialization weights
  if args.model:
    filename = os.path.splitext(os.path.basename(args.model))[0]
  else:
    filename = os.path.splitext(os.path.basename(args.weight))[0]

  tag = args.tag
  tag = tag if tag else 'default'
  filename = tag + '/' + filename

  imdb = get_imdb(args.imdb_name)
  imdb.competition_mode(args.comp_mode)

  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  #tfconfig = tf.ConfigProto(log_device_placement=True)
  tfconfig.gpu_options.allow_growth=True
  #tfconfig = tf.ConfigProto(device_count={'GPU':0})

  # init session
  sess = tf.Session(config=tfconfig)
  # load network
  if args.net == 'vgg16':
    net = vgg16(batch_size=1)
  elif args.net == 'res50':
    net = resnetv1(batch_size=1, num_layers=50)
  elif args.net == 'res101':
    net = resnetv1(batch_size=1, num_layers=101)
  elif args.net == 'res152':
    net = resnetv1(batch_size=1, num_layers=152)
  elif args.net == 'mobile':
    net = mobilenetv1(batch_size=1)
  else:
    raise NotImplementedError

  # load model
  net.create_architecture(sess, "TEST", imdb.num_classes, tag='default',
                          anchor_scales=cfg.ANCHOR_SCALES,
                          anchor_ratios=cfg.ANCHOR_RATIOS)

  if args.model:
    print(('Loading model check point from {:s}').format(args.model))
    saver = tf.train.Saver()
    saver.restore(sess, args.model)
    print('Loaded.')
  else:
    print(('Loading initial weights from {:s}').format(args.weight))
    sess.run(tf.global_variables_initializer())
    print('Loaded.')
  #pdb.set_trace() 
  #conf_threshs = [0, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.96, 0.97, 0.98, 0.99, 0.995]
  conf_threshs = [0.5]
  test_file = open('test_result_person.txt','w')
  for conf_thresh in conf_threshs:
      test_net(sess, net, imdb, filename, test_file, max_per_image=args.max_per_image, thresh=conf_thresh)
      test_file.write('\n')
  test_file.close()

  sess.close()
