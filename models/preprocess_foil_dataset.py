import csv
import json
from argparse import ArgumentParser

import en_core_web_sm

from utils import Progbar

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_filename", type=str, required=True)
    parser.add_argument("--preprocessed_dataset_filename", type=str, required=True)
    args = parser.parse_args()

    nlp = en_core_web_sm.load()
    num_lines = len([1 for line in open(args.dataset_filename)])

    images = {}

    with open(args.dataset_filename) as in_file:
        dataset = json.load(in_file)

        for num_image, image in enumerate(dataset["images"], 1):
            images[image["id"]] = image["file_name"]

        print("Found {} images".format(len(images)))
        with open(args.preprocessed_dataset_filename, mode="w") as out_file:
            writer = csv.writer(out_file, delimiter="\t")

            progress = Progbar(len(dataset["annotations"]))
            for num_annotation, annotation in enumerate(dataset["annotations"], 1):
                caption = annotation["caption"]
                image = images[annotation["image_id"]]
                label = "yes" if annotation["foil_word"] == "ORIG" else "no"
                caption_tokens = [token.lower_ for token in nlp(caption)]
                writer.writerow([label, " ".join(caption_tokens), image, caption])
                progress.update(num_annotation)
