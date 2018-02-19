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

    dataset = pd.read_csv(args.dataset_filename, sep="\t")
    dataset_X = dataset.ix[:, 1:]
    dataset_y = dataset.ix[:, 0]
    training_set, test_set = train_test_split(
        dataset_X,
        dataset_y,
        test_size=0.2,
        stratify=dataset_y,
        random_state=12345
    )

    training_set_X = training_set.ix[:, 1:]
    training_set_y = training_set.ix[:, 0]
    training_set, validation_set = train_test_split(
        training_set_X,
        training_set_y,
        test_size=0.2,
        stratify=training_set_y,
        random_state=12345
    )

    training_set.to_csv(args.training_set_filename, sep="\t", index=False)
    test_set.to_csv(args.test_set_filename, sep="\t", index=False)
    validation_set.to_csv(args.validation_set_filename, sep="\t", index=False)
