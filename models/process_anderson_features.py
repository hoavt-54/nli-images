import base64
import csv
import json
from argparse import ArgumentParser

import numpy as np
import sys

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bottom_up_features_filename", type=str)
    parser.add_argument("--mscoco_captions_train_filename", type=str)
    parser.add_argument("--mscoco_captions_val_filename", type=str)
    parser.add_argument("--img_names_filename", type=str)
    parser.add_argument("--img_features_filename", type=str)
    args = parser.parse_args()

    id2jpg = {}

    with open(args.mscoco_captions_train_filename) as in_file:
        mscoco_captions_train = json.load(in_file)
        for num_image, image in enumerate(mscoco_captions_train["images"], 1):
            print("Processing image {}/{}".format(num_image, len(mscoco_captions_train["images"])))
            id2jpg[image["id"]] = image["file_name"]

    with open(args.mscoco_captions_val_filename) as in_file:
        mscoco_captions_val = json.load(in_file)
        for num_image, image in enumerate(mscoco_captions_val["images"], 1):
            print("Processing image {}/{}".format(num_image, len(mscoco_captions_val["images"])))
            id2jpg[image["id"]] = image["file_name"]

    csv.field_size_limit(sys.maxsize)
    FIELDNAMES = ["image_id", "image_w", "image_h", "num_boxes", "boxes", "features"]

    img_labels = []
    img_features = []

    num_lines = len([1 for line in open(args.bottom_up_features_filename)])

    with open(args.bottom_up_features_filename, "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter="\t", fieldnames=FIELDNAMES)

        for num_item, item in enumerate(reader, 1):
            print("Processing image {}/{}".format(num_item, num_lines))
            image_id = int(item["image_id"])
            num_boxes = int(item["num_boxes"])
            image_features = np.frombuffer(
                base64.decodestring(item["features"]),
                dtype=np.float32
            ).reshape((num_boxes, -1))
            img_labels.append(id2jpg[image_id])
            img_features.append(image_features)

        img_features = np.array(img_features)

    with open(args.img_names_filename, mode="w") as out_file:
        json.dump(img_labels, out_file)

    np.save(args.img_features_filename, img_features)
