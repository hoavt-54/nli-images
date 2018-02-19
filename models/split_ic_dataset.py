from argparse import ArgumentParser

import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_filename", type=str, required=True)
    parser.add_argument("--training_set_filename", type=str, required=True)
    parser.add_argument("--test_set_filename", type=str, required=True)
    parser.add_argument("--validation_set_filename", type=str, required=True)
    args = parser.parse_args()

    dataset = pd.read_csv(args.dataset_filename, sep="\t")
    dataset_X = dataset.ix[:, 1:].values
    dataset_y = dataset.ix[:, 0].values

    X_train, X_test, y_train, y_test = train_test_split(
        dataset_X,
        dataset_y,
        test_size=0.2,
        stratify=dataset_y,
        random_state=12345
    )

    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=12345
    )

    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)
    y_dev = y_dev.reshape(y_dev.shape[0], 1)

    training_set = np.concatenate((y_train, X_train), axis=1)
    test_set = np.concatenate((y_test, X_test), axis=1)
    validation_set = np.concatenate((y_dev, X_dev), axis=1)

    training_set = pd.DataFrame(training_set)
    test_set = pd.DataFrame(test_set)
    validation_set = pd.DataFrame(validation_set)

    training_set.to_csv(args.training_set_filename, sep="\t", index=False, header=False)
    test_set.to_csv(args.test_set_filename, sep="\t", index=False, header=False)
    validation_set.to_csv(args.validation_set_filename, sep="\t", index=False, header=False)
