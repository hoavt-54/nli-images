import csv
from argparse import ArgumentParser

import en_core_web_sm

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_filename", type=str, required=True)
    parser.add_argument("--preprocessed_dataset_filename", type=str, required=True)
    args = parser.parse_args()

    nlp = en_core_web_sm.load()

    with open(args.dataset_filename) as in_file:
        reader = csv.reader(in_file, delimiter="\t")

        with open(args.preprocessed_dataset_filename, mode="w") as out_file:
            writer = csv.writer(out_file, delimiter="\t")

            for row in reader:
                label = row[0].strip()
                premise = row[1].strip()
                hypothesis = row[2].strip()
                premise_tokens = [token.lower_ for token in nlp(premise)]
                hypothesis_tokens = [token.lower_ for token in nlp(hypothesis)]
                writer.writerow([label, premise, hypothesis, " ".join(premise_tokens), " ".join(hypothesis_tokens)])
