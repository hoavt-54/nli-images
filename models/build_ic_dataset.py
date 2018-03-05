import collections
import csv
import json
from argparse import ArgumentParser

import matplotlib
import numpy as np

matplotlib.use("Agg")

from pycocotools.coco import COCO


def get_num_overlapping_cats(image_a_id, image_b_id, a_coco_instances_reader, b_coco_instances_reader):
    image_a_instances = a_coco_instances_reader.loadAnns(a_coco_instances_reader.getAnnIds(imgIds=image_a_id))
    image_b_instances = b_coco_instances_reader.loadAnns(b_coco_instances_reader.getAnnIds(imgIds=image_b_id))

    image_a_cats = set([a_coco_instances_reader.loadCats(i["category_id"])[0]["name"] for i in image_a_instances])
    image_b_cats = set([b_coco_instances_reader.loadCats(i["category_id"])[0]["name"] for i in image_b_instances])

    return len(image_a_cats.intersection(image_b_cats))


if __name__ == "__main__":
    np.random.seed(12345)
    parser = ArgumentParser()
    parser.add_argument("--foil_filename", type=str, required=True)
    parser.add_argument("--mscoco_captions_train_filename", type=str, required=True)
    parser.add_argument("--mscoco_captions_dev_filename", type=str, required=True)
    parser.add_argument("--mscoco_instances_train_filename", type=str, required=True)
    parser.add_argument("--mscoco_instances_dev_filename", type=str, required=True)
    parser.add_argument("--ic_dataset_filename", type=str, required=True)
    parser.add_argument("--num_positives", type=int, default=2)
    parser.add_argument("--overlapping_threshold", type=int, default=2)
    args = parser.parse_args()

    foil_captions = collections.defaultdict(lambda: collections.defaultdict(list))

    with open(args.foil_train_filename) as in_file:
        foil = json.load(in_file)

        for annotation in foil["annotations"]:
            if annotation["target_word"] == "ORIG":
                foil_captions[annotation["image_id"]]["pos"].append(annotation["caption"])
            else:
                foil_captions[annotation["image_id"]]["neg"].append(annotation["caption"])

    coco_captions_train = COCO(args.mscoco_captions_train_filename)
    coco_captions_dev = COCO(args.mscoco_captions_dev_filename)
    coco_instances_train = COCO(args.mscoco_instances_train_filename)
    coco_instances_dev = COCO(args.mscoco_instances_dev_filename)
    original_coco_captions_reader = None
    original_coco_instances_reader = None
    half_num_negatives = args.num_positives // 2

    with open(args.ic_dataset_filename, mode="w") as out_file:
        writer = csv.writer(out_file, delimiter="\t")

        for image_number, image_id in enumerate(foil_captions, 1):
            print("[{}/{}] Processing image {}".format(image_number, len(foil_captions), image_id))

            if coco_captions_train.getAnnIds(imgIds=image_id):
                original_coco_captions_reader = coco_captions_train
                original_coco_instances_reader = coco_instances_train
            elif coco_captions_dev.getAnnIds(imgIds=image_id):
                original_coco_captions_reader = coco_captions_dev
                original_coco_instances_reader = coco_instances_dev
            else:
                print("{} not found!".format(image_id))

            image_filename = original_coco_captions_reader.loadImgs(image_id)[0]["file_name"]
            captions_ids = original_coco_captions_reader.getAnnIds(imgIds=image_id)
            sampled_pos_captions_ids = np.random.choice(captions_ids, size=args.num_positives)
            sampled_pos_captions = [c["caption"] for c in original_coco_captions_reader.loadAnns(sampled_pos_captions_ids)]
            sampled_foil_neg_captions = list(np.random.choice(foil_captions[image_id]["neg"], size=half_num_negatives))
            sampled_foil_neg_images = [image_filename] * half_num_negatives
            sampled_mscoco_neg_captions = []
            sampled_mscoco_neg_images = []
            neg_image_ids = original_coco_captions_reader.getImgIds() + original_coco_captions_reader.getImgIds()
            neg_image_ids.remove(image_id)

            for i in range(half_num_negatives):
                sampled_neg_image_id = np.random.choice(neg_image_ids)
                print("Sampled image {}".format(sampled_neg_image_id))

                sampled_coco_instances_reader = None

                if coco_captions_train.getAnnIds(imgIds=sampled_neg_image_id):
                    sampled_coco_captions_reader = coco_captions_train
                    sampled_coco_instances_reader = coco_instances_train
                elif coco_captions_dev.getAnnIds(imgIds=sampled_neg_image_id):
                    sampled_coco_captions_reader = coco_captions_dev
                    sampled_coco_instances_reader = coco_instances_dev
                else:
                    print("{} not found!".format(sampled_neg_image_id))

                while get_num_overlapping_cats(image_id,
                                               sampled_neg_image_id,
                                               original_coco_instances_reader,
                                               sampled_coco_instances_reader
                                               ) > args.overlapping_threshold:
                    sampled_neg_image_id = np.random.choice(neg_image_ids)
                    print("Sampled image {}".format(sampled_neg_image_id))

                    if coco_captions_train.getAnnIds(imgIds=sampled_neg_image_id):
                        sampled_coco_captions_reader = coco_captions_train
                        sampled_coco_instances_reader = coco_instances_train
                    elif coco_captions_dev.getAnnIds(imgIds=sampled_neg_image_id):
                        sampled_coco_captions_reader = coco_captions_dev
                        sampled_coco_instances_reader = coco_instances_dev
                    else:
                        print("{} not found!".format(sampled_neg_image_id))

                captions_ids = sampled_coco_captions_reader.getAnnIds(imgIds=sampled_neg_image_id)
                sampled_neg_caption_id = np.random.choice(captions_ids)
                sampled_mscoco_neg_captions.append(sampled_coco_captions_reader.loadAnns(int(sampled_neg_caption_id))[0]["caption"])
                sampled_mscoco_neg_images.append(sampled_coco_captions_reader.loadImgs(int(sampled_neg_image_id))[0]["file_name"])

            for pos_caption in sampled_pos_captions:
                writer.writerow(["yes", pos_caption, image_filename, "mscoco"])

            for neg_caption, neg_image_filename in zip(sampled_foil_neg_captions, sampled_foil_neg_images):
                writer.writerow(["no", neg_caption, neg_image_filename, "foil"])

            for neg_caption, neg_image_filename in zip(sampled_mscoco_neg_captions, sampled_mscoco_neg_images):
                writer.writerow(["no", neg_caption, neg_image_filename, "mscoco"])
