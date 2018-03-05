from argparse import ArgumentParser

import pandas as pd

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--sick2_filename", type=str, required=True)
    parser.add_argument("--difficult_sick2_filename", type=str, required=True)
    parser.add_argument("--flickr_filename", type=str, required=True)
    parser.add_argument("--vsick2_filename", type=str, required=True)
    parser.add_argument("--difficult_vsick2_filename", type=str, required=True)
    args = args = parser.parse_args()

    sick2 = pd.read_csv(args.sick2_filename, sep="\t")
    difficult_sick2 = pd.read_csv(args.difficult_sick2_filename, sep="\t")
    flickr = pd.read_csv(args.flickr_filename, sep="\t", header=None)

    sentences_A = []
    for index, row in sick2.iterrows():
        joined_A = "".join([x for x in sick2.sentence_A_original[index].lower() if x.isalpha()])
        sentences_A.append(joined_A)
    sick2["sent_A"] = pd.Series(sentences_A, index=sick2.index)

    flickr_joined = []
    for index, row in flickr.iterrows():
        flickr_joined.append("".join([x for x in flickr.iloc[index][1].lower() if x.isalpha()]))
    flickr["joined"] = pd.Series(flickr_joined, index=flickr.index)

    vsick2 = sick2.merge(flickr, left_on="sent_A", right_on="joined", how="inner")
    del vsick2[1]
    del vsick2["sent_A"]
    del vsick2["joined"]
    vsick2.rename(columns={0: "jpg"}, inplace=True)
    vsick2.reset_index(inplace=True)
    vsick2.to_csv(args.vsick2_filename, sep="\t", index=None)

    difficult_sset_ind = difficult_sick2.query("ENT_difficult == 1 | REL_difficult == 1")
    difficult_vsick2 = vsick2[vsick2["pair_ID"].isin(difficult_sset_ind.pair_ID)]
    difficult_vsick2.to_csv(args.difficult_vsick2_filename, index=None, sep="\t")
