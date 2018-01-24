#!/usr/bin/env python
import json
from argparse import ArgumentParser

import _init_paths
from fast_rcnn.config import cfg, cfg_from_file
from fast_rcnn.test import im_detect, _get_blobs
from fast_rcnn.nms_wrapper import nms
import caffe
import cv2
import numpy as np
import os
import pickle


def get_detections_from_im(net, im_file, image_id, conf_thresh=0.2, min_num_boxes=36, max_num_boxes=36):
    im = cv2.imread(im_file)
    scores, boxes, attr_scores, rel_scores = im_detect(net, im)

    # keep the original boxes, don't worry about the regression bounding box outputs
    rois = net.blobs["rois"].data.copy()

    # unscale back to the raw image space
    blobs, im_scales = _get_blobs(im, None)
    cls_boxes = rois[:, 1:5] / im_scales[0]
    cls_prob = net.blobs["cls_prob"].data
    pool5 = net.blobs["pool5_flat"].data

    # keep only the best detections
    max_conf = np.zeros((rois.shape[0]))
    for cls_ind in range(1, cls_prob.shape[1]):
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])).astype(np.float32)
        keep = np.array(nms(dets, cfg.TEST.NMS))
        max_conf[keep] = np.where(cls_scores[keep] > max_conf[keep], cls_scores[keep], max_conf[keep])

    keep_boxes = np.where(max_conf >= conf_thresh)[0]
    if len(keep_boxes) < min_num_boxes:
        keep_boxes = np.argsort(max_conf)[::-1][:min_num_boxes]
    elif len(keep_boxes) > max_num_boxes:
        keep_boxes = np.argsort(max_conf)[::-1][:max_num_boxes]

    return {
        "image_h": np.size(im, 0),
        "image_w": np.size(im, 1),
        "num_boxes": len(keep_boxes),
        "boxes": cls_boxes[keep_boxes],
        "features": pool5[keep_boxes]
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cfg_filename", type=str,
                        default="../experiments/cfgs/faster_rcnn_end2end_resnet.yml")
    parser.add_argument("--net_filename", type=str,
                        default="../data/faster_rcnn_models/resnet101_faster_rcnn_final_iter_320000.caffemodel")
    parser.add_argument("--def_filename", type=str,
                        default="../models/vg/ResNet-101/faster_rcnn_end2end_final/test.prototxt")
    parser.add_argument("--img_path", type=str, required=True)
    parser.add_argument("--num_boxes", type=int, default=36)
    parser.add_argument("--img_extension", type=str, default=".jpg")
    parser.add_argument("--out_filename", type=str, required=True)
    args = parser.parse_args()

    gpu_id = 0
    caffe.set_device(gpu_id)
    caffe.set_mode_gpu()
    net = None
    cfg_from_file(args.cfg_filename)
    net = caffe.Net(args.def_filename, caffe.TEST, weights=args.net_filename)

    bottom_up_features = {}
    for filename in os.listdir(args.img_path):
        if filename.endswith(args.img_extension):
            full_filename = os.path.join(args.img_path, filename)
            print("Processing file: {}".format(full_filename))
            results = get_detections_from_im(
                net,
                full_filename,
                filename,
                conf_thresh=0.2,
                min_num_boxes=args.num_boxes,
                max_num_boxes=args.num_boxes
            )
            bottom_up_features[filename] = results

    with open(args.out_filename, mode="wb") as out_file:
        pickle.dump(bottom_up_features, out_file)
