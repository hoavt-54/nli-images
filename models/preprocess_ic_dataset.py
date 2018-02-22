import csv
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

    with open(args.dataset_filename) as in_file:
        reader = csv.reader(in_file, delimiter="\t")

        with open(args.preprocessed_dataset_filename, mode="w") as out_file:
            writer = csv.writer(out_file, delimiter="\t")

            progress = Progbar(num_lines)
            for row_number, row in enumerate(reader, 1):
                progress.update(row_number)
                label = row[0].strip()
                caption = row[1].strip()
                image_filename = row[2].strip()
                source = row[3].strip()
                caption_tokens = [token.lower_ for token in nlp(caption)]
                writer.writerow([label, " ".join(caption_tokens), image_filename, source, caption])
