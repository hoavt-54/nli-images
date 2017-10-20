from argparse import ArgumentParser

import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--full_sick_filename", type=str, required=True)
    parser.add_argument("--train_sick_filename", type=str, required=True)
    parser.add_argument("--dev_sick_filename", type=str, required=True)
    parser.add_argument("--test_sick_filename", type=str, required=True)
    args = args = parser.parse_args()

    full_sick = pd.read_csv(args.full_sick_filename, sep="\t")

    full_sick_train = full_sick.query("SemEval_set == 'TRAIN'")
    full_sick_train.to_csv(args.train_sick_filename, sep="\t", index=False)

    full_sick_dev = full_sick.query("SemEval_set == 'TRIAL'")
    full_sick_dev.to_csv(args.dev_sick_filename, sep="\t", index=False)

    full_sick_test = full_sick.query("SemEval_set == 'TEST'")
    full_sick_test.to_csv(args.test_sick_filename, sep="\t", index=False)
