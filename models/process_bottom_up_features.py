import json
import pickle
from argparse import ArgumentParser

import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bottom_up_features_filename", type=str)
    parser.add_argument("--img_names_filename", type=str)
    parser.add_argument("--img_features_filename", type=str)
    args = parser.parse_args()

    img_labels = []
    img_features = []

    with open(args.bottom_up_features_filename, mode="rb") as in_file:
        bottom_up_features = pickle.load(in_file, encoding="latin1")

        for label, features in bottom_up_features.items():
            img_labels.append(label)
            img_features.append(features)

    img_features = np.array(img_features)

    with open(args.img_names_filename, mode="w") as out_file:
        json.dump(img_labels, out_file)

    np.save(args.img_features_filename, img_features)
