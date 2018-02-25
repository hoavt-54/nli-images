import csv
import json
from argparse import ArgumentParser

from utils import Progbar


def extract_tokens_from_binary_parse(parse):
    return parse.replace("(", " ").replace(")", " ").replace("-LRB-", "(").replace("-RRB-", ")").split()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_filename", type=str, required=True)
    parser.add_argument("--filtered_dataset_filename", type=str, required=True)
    parser.add_argument("--preprocessed_dataset_filename", type=str, required=True)
    args = parser.parse_args()

    num_lines = len([1 for line in open(args.dataset_filename)])
    progress = Progbar(num_lines)

    available_images = set()
    with open(args.filtered_dataset_filename) as in_file:
        reader = csv.reader(in_file, delimiter="\t")
        for row in reader:
            available_images.add(row[3].strip())

    with open(args.dataset_filename) as in_file:
        with open(args.preprocessed_dataset_filename, mode="w") as out_file:
            writer = csv.writer(out_file, delimiter="\t")

            for row_number, line in enumerate(in_file, 1):
                data = json.loads(line)

                premise = data["sentence1"]
                hypothesis = data["sentence2"]
                premise_tokens = extract_tokens_from_binary_parse(data["sentence1_binary_parse"])
                hypothesis_tokens = extract_tokens_from_binary_parse(data["sentence2_binary_parse"])
                label = data["gold_label"]
                image_filename = data["captionID"].split("#")[0]
                if image_filename in available_images and label != "-":
                    writer.writerow([label, " ".join(premise_tokens), " ".join(hypothesis_tokens), image_filename, premise, hypothesis])
                progress.update(row_number)
