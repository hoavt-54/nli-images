from argparse import ArgumentParser

import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sick_filename", type=str, required=True)
    parser.add_argument("--difficult_sick_filename", type=str, required=True)
    parser.add_argument("--filtered_sick_filename", type=str, required=True)
    parser.add_argument("--difficult_filtered_sick_filename", type=str, required=True)
    args = args = parser.parse_args()

    full_sick = pd.read_csv(args.sick_filename, sep="\t")
    difficult_sick = pd.read_csv(args.difficult_sick_filename, sep="\t")

    filtered_sick = full_sick.query("sentence_A_expRule == 'S1_null' and sentence_A_dataset == 'FLICKR'")
    filtered_sick.to_csv(args.filtered_sick_filename, sep="\t", index=False)

    difficult_sset_ind = difficult_sick.query("ENT_difficult == 1 | REL_difficult == 1")
    difficult_filtered_sick = filtered_sick[filtered_sick["pair_ID"].isin(difficult_sset_ind.pair_ID)]
    difficult_filtered_sick.to_csv(args.difficult_filtered_sick_filename, index=None, sep="\t")
