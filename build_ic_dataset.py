import collections
import csv
import json
from argparse import ArgumentParser

import numpy as np
from pycocotools.coco import COCO


def get_num_overlapping_cats(a, b):
    return 1


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--foil_filename", type=str, required=True)
    parser.add_argument("--mscoco_train_filename", type=str, required=True)
    parser.add_argument("--mscoco_dev_filename", type=str, required=True)
    parser.add_argument("--ic_dataset_filename", type=str, required=True)
    parser.add_argument("--num_positives", type=int, default=2)
    parser.add_argument("--overlapping_threshold", type=int, default=2)
    args = parser.parse_args()

    foil_captions = collections.defaultdict(lambda: collections.defaultdict(list))

    with open(args.foil_filename) as in_file:
        foil = json.load(in_file)

        for annotation in foil["annotations"]:
            if annotation["target_word"] == "ORIG":
                foil_captions[annotation["image_id"]]["pos"].append(annotation["caption"])
            else:
                foil_captions[annotation["image_id"]]["neg"].append(annotation["caption"])

    coco_captions_train = COCO(args.mscoco_train_filename)
    coco_captions_dev = COCO(args.mscoco_dev_filename)
    coco_captions = None
    half_num_negatives = args.num_positives // 2

    with open(args.ic_dataset_filename, mode="w") as out_file:
        writer = csv.writer(out_file, delimiter="\t")

        for image_id in list(foil_captions)[:10]:
            print("Processing image {}".format(image_id))

            if coco_captions_train.getAnnIds(imgIds=image_id):
                coco_captions = coco_captions_train
            elif coco_captions_train.getAnnIds(imgIds=image_id):
                coco_captions = coco_captions_dev
            else:
                print("{} not found!".format(image_id))

            image_filename = coco_captions.loadImgs(image_id)[0]["file_name"]
            captions_ids = coco_captions.getAnnIds(imgIds=image_id)
            sampled_pos_captions_ids = np.random.choice(captions_ids, size=args.num_positives)
            sampled_pos_captions = [c["caption"] for c in coco_captions.loadAnns(sampled_pos_captions_ids)]
            sampled_neg_captions = list(np.random.choice(foil_captions[image_id]["neg"], size=half_num_negatives))
            sampled_neg_images = [image_filename] * half_num_negatives
            neg_image_ids = list(foil_captions.keys())
            neg_image_ids.remove(image_id)

            for i in range(half_num_negatives):
                sampled_neg_image_id = np.random.choice(neg_image_ids)
                print("Sampled image {}".format(sampled_neg_image_id))

                while get_num_overlapping_cats(image_id, sampled_neg_image_id) > args.overlapping_threshold:
                    sampled_neg_image_id = np.random.choice(neg_image_ids)
                    print("Sampled image {}".format(sampled_neg_image_id))

                captions_ids = coco_captions.getAnnIds(imgIds=sampled_neg_image_id)
                sampled_neg_caption_id = np.random.choice(captions_ids)
                sampled_neg_captions.append(coco_captions.loadAnns(int(sampled_neg_caption_id))[0]["caption"])
                sampled_neg_images.append(coco_captions.loadImgs(int(sampled_neg_image_id))[0]["file_name"])

            for pos_caption in sampled_pos_captions:
                writer.writerow(["yes", pos_caption, image_filename])

            for neg_caption, neg_image_filename in zip(sampled_neg_captions, sampled_neg_images):
                writer.writerow(["no", neg_caption, neg_image_filename])
