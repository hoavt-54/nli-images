from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_filename", type=str, required=True)
    parser.add_argument("--training_set_filename", type=str, required=True)
    parser.add_argument("--test_set_filename", type=str, required=True)
    parser.add_argument("--validation_set_filename", type=str, required=True)
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset_filename)
    training_set, test_set = train_test_split(dataset, test_size=0.2, random_state=12345)
    training_set, validation_set = train_test_split(training_set, test_size=0.2, random_state=12345)

    training_set.to_csv(args.training_set_filename, sep="\t", index=False)
    test_set.to_csv(args.test_set_filename, sep="\t", index=False)
    validation_set.to_csv(args.validation_set_filename, sep="\t", index=False)
