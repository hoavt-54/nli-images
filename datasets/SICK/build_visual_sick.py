from argparse import ArgumentParser

import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--filtered_sick_filename", type=str, required=True)
    parser.add_argument("--difficult_sick_filename", type=str, required=True)
    parser.add_argument("--flickr_filename", type=str, required=True)
    parser.add_argument("--visual_sick_filename", type=str, required=True)
    parser.add_argument("--difficult_visual_sick_filename", type=str, required=True)
    args = args = parser.parse_args()

    filtered_sick = pd.read_csv(args.filtered_sick_filename, sep="\t")
    difficult_sick = pd.read_csv(args.difficult_sick_filename, sep="\t")
    flickr = pd.read_csv(args.flickr_filename, sep="\t", header=None)

    sentences_A = []
    sentences_B = []

    for index, row in filtered_sick.iterrows():
        joined_A = "".join([x for x in filtered_sick.sentence_A_original[index].lower() if x.isalpha()])
        joined_B = "".join([x for x in filtered_sick.sentence_B_original[index].lower() if x.isalpha()])
        sentences_A.append(joined_A)
        sentences_B.append(joined_B)

    filtered_sick["sent_A"] = pd.Series(sentences_A, index=filtered_sick.index)

    flickr_joined = []

    for index, row in flickr.iterrows():
        flickr_joined.append("".join([x for x in flickr.iloc[index][1].lower() if x.isalpha()]))

    flickr["joined"] = pd.Series(flickr_joined, index=flickr.index)

    visual_sick = filtered_sick.merge(flickr, left_on="sent_A", right_on="joined", how="inner")
    del visual_sick[1]
    del visual_sick["sent_A"]
    del visual_sick["joined"]
    visual_sick.rename(columns={0: "jpg"}, inplace=True)
    visual_sick.reset_index(inplace=True)
    visual_sick.to_csv(args.visual_sick_filename, sep="\t", index=None)

    difficult_sset_ind = difficult_sick.query("ENT_difficult == 1 | REL_difficult == 1")
    difficult_visual_sick = visual_sick[visual_sick["pair_ID"].isin(difficult_sset_ind.pair_ID)]
    difficult_visual_sick.to_csv(args.difficult_visual_sick_filename, index=None, sep="\t")
