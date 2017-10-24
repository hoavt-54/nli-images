from argparse import ArgumentParser
import pandas as pd
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--filename", type=str, required=True)
    parser.add_argument("--converted_filename", type=str, required=True)
    args = args = parser.parse_args()
    sick = pd.read_csv(args.filename, sep="\t", columns="")
    converted_sick = sick[["entailment_label", "sentence_A", "sentence_B"]]
    converted_sick["jpg"] = "placeholder"
    converted_sick.to_csv(args.converted_filename, sep="\t", header=False, index=False)
