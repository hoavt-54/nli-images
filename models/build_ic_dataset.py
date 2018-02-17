import collections
import csv
import json
from argparse import ArgumentParser

import numpy as np
from pycocotools.coco import COCO


def get_num_overlapping_cats(image_a_id, image_b_id, coco_instances):
    image_a_instances = coco_instances.loadAnns(coco_instances.getAnnIds(imgIds=image_a_id))
    image_b_instances = coco_instances.loadAnns(coco_instances.getAnnIds(imgIds=image_b_id))

    image_a_cats = set([coco_instances.loadCats(i["category_id"])[0]["name"] for i in image_a_instances])
    image_b_cats = set([coco_instances.loadCats(i["category_id"])[0]["name"] for i in image_b_instances])

    return len(image_a_cats.intersection(image_b_cats))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--foil_train_filename", type=str, required=True)
    parser.add_argument("--foil_test_filename", type=str, required=True)
    parser.add_argument("--mscoco_captions_train_filename", type=str, required=True)
    parser.add_argument("--mscoco_captions_dev_filename", type=str, required=True)
    parser.add_argument("--mscoco_instances_train_filename", type=str, required=True)
    parser.add_argument("--mscoco_instances_dev_filename", type=str, required=True)
    parser.add_argument("--ic_dataset_filename", type=str, required=True)
    parser.add_argument("--num_positives", type=int, default=2)
    parser.add_argument("--overlapping_threshold", type=int, default=2)
    args = parser.parse_args()

    np.random.seed(12345)
    foil_captions = collections.defaultdict(lambda: collections.defaultdict(list))

    with open(args.foil_train_filename) as in_file:
        foil = json.load(in_file)

        for annotation in foil["annotations"]:
            if annotation["target_word"] == "ORIG":
                foil_captions[annotation["image_id"]]["pos"].append(annotation["caption"])
            else:
                foil_captions[annotation["image_id"]]["neg"].append(annotation["caption"])

    with open(args.foil_test_filename) as in_file:
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
    coco_captions = None
    coco_instances = None
    half_num_negatives = args.num_positives // 2

    with open(args.ic_dataset_filename, mode="w") as out_file:
        writer = csv.writer(out_file, delimiter="\t")

        for image_number, image_id in enumerate(foil_captions, 1):
            print("[{}/{}] Processing image {}".format(image_number, len(foil_captions), image_id))

            if coco_captions_train.getAnnIds(imgIds=image_id):
                coco_captions = coco_captions_train
                coco_instances = coco_instances_train
            elif coco_captions_train.getAnnIds(imgIds=image_id):
                coco_captions = coco_captions_dev
                coco_instances = coco_instances_dev
            else:
                print("{} not found!".format(image_id))

            image_filename = coco_captions.loadImgs(image_id)[0]["file_name"]
            captions_ids = coco_captions.getAnnIds(imgIds=image_id)
            sampled_pos_captions_ids = np.random.choice(captions_ids, size=args.num_positives)
            sampled_pos_captions = [c["caption"] for c in coco_captions.loadAnns(sampled_pos_captions_ids)]
            sampled_foil_neg_captions = list(np.random.choice(foil_captions[image_id]["neg"], size=half_num_negatives))
            sampled_foil_neg_images = [image_filename] * half_num_negatives
            sampled_mscoco_neg_captions = []
            sampled_mscoco_neg_images = []
            neg_image_ids = list(foil_captions.keys())
            neg_image_ids.remove(image_id)

            for i in range(half_num_negatives):
                sampled_neg_image_id = np.random.choice(neg_image_ids)
                print("Sampled image {}".format(sampled_neg_image_id))

                while get_num_overlapping_cats(image_id, sampled_neg_image_id, coco_instances)\
                        > args.overlapping_threshold:
                    sampled_neg_image_id = np.random.choice(neg_image_ids)
                    print("Sampled image {}".format(sampled_neg_image_id))

                captions_ids = coco_captions.getAnnIds(imgIds=sampled_neg_image_id)
                sampled_neg_caption_id = np.random.choice(captions_ids)
                sampled_mscoco_neg_captions.append(coco_captions.loadAnns(int(sampled_neg_caption_id))[0]["caption"])
                sampled_mscoco_neg_images.append(coco_captions.loadImgs(int(sampled_neg_image_id))[0]["file_name"])

            for pos_caption in sampled_pos_captions:
                writer.writerow(["yes", pos_caption, image_filename, "mscoco"])

            for neg_caption, neg_image_filename in zip(sampled_foil_neg_captions, sampled_foil_neg_images):
                writer.writerow(["no", neg_caption, neg_image_filename, "foil"])

            for neg_caption, neg_image_filename in zip(sampled_mscoco_neg_captions, sampled_mscoco_neg_images):
                writer.writerow(["no", neg_caption, neg_image_filename, "mscoco"])
