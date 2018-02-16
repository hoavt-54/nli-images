import collections
import json
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--foil_filename", type=str, required=True)
    parser.add_argument("--mscoco_filename", type=str, required=False)
    args = parser.parse_args()

    foil_captions = collections.defaultdict(lambda: collections.defaultdict(list))

    with open(args.foil_filename) as in_file:
        foil = json.load(in_file)

        for annotation in foil["annotations"]:
            if annotation["target_word"] == "ORIG":
                foil_captions[annotation["image_id"]]["pos"].append(annotation["caption"])
            else:
                foil_captions[annotation["image_id"]]["neg"].append(annotation["caption"])
