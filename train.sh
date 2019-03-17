#!/bin/bash

cd faster_rcnn/
CUDA_VISIBLE_DEVICES=1 python ./tools/trainval_net.py --imdb eelab_train --weight ./data/imagenet_weights/vgg16.ckpt --cfg ./experiments/cfgs/vgg16.yml --net vgg16 --iter 9500

