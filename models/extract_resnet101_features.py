#!/usr/bin/env python

import json
import os
from argparse import ArgumentParser
from utils import Progbar

import _init_paths
import caffe
import cv2
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--net_weights_filename", type=str, default="../data/imagenet_models/ResNet-101-model.caffemodel")
    parser.add_argument("--net_def_filename", type=str, default="../data/imagenet_models/ResNet-101-deploy.prototxt")
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--img_names_filename", type=str, required=True)
    parser.add_argument("--img_features_filename", type=str, required=True)
    args = parser.parse_args()

    net = caffe.Net(args.net_def_filename, args.net_weights_filename, caffe.TEST)

    img_filenames = os.listdir(args.img_path)
    num_img_filenames = len(img_filenames)
    img_features = []
    progress = Progbar(num_img_filenames)

    for num_filename, filename in enumerate(img_filenames, 1):
        im = cv2.resize(cv2.imread(os.path.join(args.img_path, filename)), (224, 224)).astype(np.float32)
        im[:, :, 0] -= 103.939
        im[:, :, 1] -= 116.779
        im[:, :, 2] -= 123.68
        im = np.expand_dims(im, axis=0)
        im = np.transpose(im, (0, 3, 1, 2))
        net.forward(data=im)
        pool5_features = net.blobs["pool5"].data.squeeze()
        img_features.append(pool5_features)
        progress.update(num_filename)

    img_features = np.array(img_features)

    with open(args.img_names_filename, "w") as out_file:
        json.dump(img_filenames, out_file)

    np.save(args.img_features_filename, img_features)
