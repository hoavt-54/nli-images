import csv
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--training_set", type=str, required=True)
    parser.add_argument("--validation_set", type=str, required=True)
    parser.add_argument("--ic_dataset_filename", type=str, required=True)
    args = parser.parse_args()

    with open(args.ic_dataset_filename, mode="w") as out_file:
        writer = csv.writer(out_file, delimiter="\t")

        with open(args.training_set) as training_set_in_file:
            training_set_reader = csv.reader(training_set_in_file, delimiter="\t")

        with open(args.validation_set) as validation_set_in_file:
            validation_set_reader = csv.reader(validation_set_in_file, delimiter="\t")

        for row in training_set_reader:
            writer.writerow(row)

        for row in validation_set_reader:
            writer.writerow(row)
