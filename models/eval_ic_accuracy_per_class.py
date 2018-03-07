from argparse import ArgumentParser

import pandas as pd
from sklearn.metrics import accuracy_score

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--predictions_filename", type=str, required=True)
    args = parser.parse_args()

    data = pd.read_csv(
        args.predictions_filename,
        sep="\t",
        header=None,
        names=["gold_label", "prediction", "sentence_tokens", "jpg", "sentence"]
    )

    print("Overall accuracy: {}".format(accuracy_score(data["gold_label"], data["prediction"])))

    data_yes = data.loc[data["gold_label"] == "yes"]
    print("Accuracy for 'yes': {}".format(accuracy_score(data_yes["gold_label"], data_yes["prediction"])))

    data_no = data.loc[data["gold_label"] == "no"]
    print("Acuracy for 'no': {}".format(accuracy_score(data_no["gold_label"], data_no["prediction"])))
