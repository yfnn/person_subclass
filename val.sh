#!/bin/bash

cd faster_rcnn/
CUDA_VISIBLE_DEVICES=1 python ./tools/test_net.py --imdb eelab_test --model ./output/vgg16/eelab_train/default/vgg16_faster_rcnn_iter_9500.ckpt --cfg ./experiments/cfgs/vgg16.yml --net vgg16
