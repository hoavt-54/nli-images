import os
import pickle
from argparse import ArgumentParser
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import pylab

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--bottom_up_features_filename", type=str)
    parser.add_argument("--img_path", type=str)
    parser.add_argument("--annotated_img_path", type=str)
    args = parser.parse_args()

    with open(args.bottom_up_features_filename, mode="rb") as in_file:
        bottom_up_features = pickle.load(in_file)

    for img_index, (img_name, img_features) in enumerate(bottom_up_features.items()):
        im = cv2.imread(os.path.join(args.img_path, img_name))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        plt.figure()
        plt.imshow(im)
        plt.axis("off")
        for i in range(img_features["num_boxes"]):
            bbox = img_features["boxes"][i]["coordinates"]
            if bbox[0] == 0:
                bbox[0] = 1
            if bbox[1] == 0:
                bbox[1] = 1
            plt.gca().add_patch(
                plt.Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1], fill=False,
                    edgecolor="red", linewidth=2, alpha=0.5
                )
            )
        print("[{}/{}] Saving to: {}".format(
            img_index + 1,
            len(bottom_up_features),
            os.path.join(args.annotated_img_path, img_name).replace(".jpg", ".png"))
        )
        pylab.savefig(os.path.join(args.annotated_img_path, img_name).replace(".jpg", ".png"))
