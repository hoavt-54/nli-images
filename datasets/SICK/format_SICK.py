from argparse import ArgumentParser

import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sick_filename", type=str, required=True)
    parser.add_argument("--formatted_sick_filename", type=str, required=True)
    args = args = parser.parse_args()
    sick = pd.read_csv(args.sick_filename, sep="\t")
    if "jpg" in sick.columns:
        formatted_sick = sick[["entailment_label", "sentence_A", "sentence_B", "jpg"]]
        formatted_sick.is_copy = False
    else:
        formatted_sick = sick[["entailment_label", "sentence_A", "sentence_B"]]
        formatted_sick.is_copy = False
        formatted_sick["jpg"] = ["#"] * len(formatted_sick)
    formatted_sick["entailment_label"] = formatted_sick["entailment_label"].replace("ENTAILMENT", "entailment")
    formatted_sick["entailment_label"] = formatted_sick["entailment_label"].replace("CONTRADICTION", "contradiction")
    formatted_sick["entailment_label"] = formatted_sick["entailment_label"].replace("NEUTRAL", "neutral")
    formatted_sick.to_csv(args.formatted_sick_filename, sep="\t", header=False, index=False)
