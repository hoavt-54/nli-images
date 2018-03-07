import csv
from argparse import ArgumentParser

import matplotlib
import numpy as np

matplotlib.use("Agg")

from pycocotools.coco import COCO


def get_num_overlapping_cats(image_a_id, image_b_id, coco_instances):
    image_a_instances = coco_instances.loadAnns(coco_instances.getAnnIds(imgIds=image_a_id))
    image_b_instances = coco_instances.loadAnns(coco_instances.getAnnIds(imgIds=image_b_id))

    image_a_cats = set([coco_instances.loadCats(i["category_id"])[0]["name"] for i in image_a_instances])
    image_b_cats = set([coco_instances.loadCats(i["category_id"])[0]["name"] for i in image_b_instances])

    return len(image_a_cats.intersection(image_b_cats))


if __name__ == "__main__":
    np.random.seed(12345)
    parser = ArgumentParser()
    parser.add_argument("--mscoco_captions_filename", type=str, required=True)
    parser.add_argument("--mscoco_instances_filename", type=str, required=True)
    parser.add_argument("--ic_dataset_filename", type=str, required=True)
    parser.add_argument("--num_positives", type=int, default=1)
    parser.add_argument("--num_negatives", type=int, default=1)
    parser.add_argument("--overlapping_threshold", type=int, default=2)
    args = parser.parse_args()

    coco_captions = COCO(args.mscoco_captions_filename)
    coco_instances = COCO(args.mscoco_instances_filename)

    with open(args.ic_dataset_filename, mode="w") as out_file:
        writer = csv.writer(out_file, delimiter="\t")

        image_ids = coco_instances.getImgIds()

        for image_number, image_id in enumerate(image_ids, 1):
            print("[{}/{}] Processing image {}".format(image_number, len(image_ids), image_id))
            image_filename = coco_captions.loadImgs(image_id)[0]["file_name"]
            image_filename = image_filename.replace("COCO_train2014_", "").replace("COCO_val2014_", "")
            captions_ids = coco_captions.getAnnIds(imgIds=image_id)
            sampled_pos_captions_ids = np.random.choice(captions_ids, size=args.num_positives)
            sampled_pos_captions = [c["caption"] for c in coco_captions.loadAnns(sampled_pos_captions_ids)]

            neg_image_ids = coco_captions.getImgIds()
            neg_image_ids.remove(image_id)
            sampled_neg_captions_ids = []
            sampled_neg_captions = []
            sampled_neg_images = []

            for i in range(args.num_negatives):
                sampled_neg_image_id = np.random.choice(neg_image_ids)
                print("Sampled image {}".format(sampled_neg_image_id))

                while get_num_overlapping_cats(image_id, sampled_neg_image_id, coco_instances) > args.overlapping_threshold:
                    sampled_neg_image_id = np.random.choice(neg_image_ids)
                    print("Sampled image {}".format(sampled_neg_image_id))

                neg_captions_ids = coco_captions.getAnnIds(imgIds=sampled_neg_image_id)
                sampled_neg_caption_id = np.random.choice(neg_captions_ids)
                sampled_neg_captions_ids.append(sampled_neg_caption_id)
                sampled_neg_captions.append(coco_captions.loadAnns(sampled_neg_caption_id)[0]["caption"])

            for pos_caption, pos_caption_id in zip(sampled_pos_captions, sampled_pos_captions_ids):
                writer.writerow(["yes", pos_caption, image_filename, pos_caption_id, image_id])

            for neg_caption, neg_caption_id in zip(sampled_neg_captions, sampled_neg_captions_ids):
                writer.writerow(["no", neg_caption, image_filename, neg_caption_id, image_id])
