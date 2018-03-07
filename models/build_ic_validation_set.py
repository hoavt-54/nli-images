import csv
from argparse import ArgumentParser
from collections import defaultdict

import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ic_validation_set_filename", type=str, required=True)
    parser.add_argument("--ic_generated_test_set_filename", type=str, required=True)
    parser.add_argument("--ic_generated_validation_set_filename", type=str, required=True)
    args = parser.parse_args()

    with open(args.ic_test_set_filename) as in_file:
        reader = csv.reader(in_file)

        image2rows = defaultdict(list)

        for row in reader:
            image_filename = row[2].strip()
            image2rows[image_filename].append(row)

        images = set(image2rows.keys())
        test_set_images = set(np.random.choice(images, size=len(images) // 2, replace=False))
        dev_set_images = images - test_set_images

    with open(args.ic_generated_test_set_filename, mode="w") as out_file:
        writer = csv.writer(out_file)

        for image in test_set_images:
            for row in image2rows[image]:
                writer.writerow(row)

    with open(args.ic_generated_dev_set_filename, mode="w") as out_file:
        writer = csv.writer(out_file)

        for image in dev_set_images:
            for row in image2rows[image]:
                writer.writerow(row)
