import csv
import json

import numpy as np

from embedding import load_glove
from preprocessing import pad_sequences


def load_te_dataset(filename, token2id, label2id):
    labels = []
    premises = []
    hypotheses = []

    with open(filename) as in_file:
        reader = csv.reader(in_file, delimiter="\t")

        for row in reader:
            labels.append(label2id[row[0].strip()])
            premise_tokens = row[1].strip().lower().split()
            hypothesis_tokens = row[2].strip().lower().split()
            num_unk_tokens = 0
            for token in premise_tokens:
                if token in token2id:
                    premises.append(token2id[token])
                else:
                    premises.append(token2id["#unk#"])
                    num_unk_tokens
            premises.append([token2id.get(token, token2id["#unk#"]) for token in premise_tokens])
            hypotheses.append([token2id.get(token, token2id["#unk#"]) for token in hypothesis_tokens])

        premises = pad_sequences(premises, padding="post", value=token2id["#pad#"], dtype=np.long)
        hypotheses = pad_sequences(hypotheses, padding="post", value=token2id["#pad#"], dtype=np.long)
        labels = np.array(labels)

        return labels, premises, hypotheses


def load_vte_dataset(nli_dataset_filename, token2id, label2id):
    labels = []
    premises = []
    hypotheses = []
    img_names = []

    with open(nli_dataset_filename) as in_file:
        reader = csv.reader(in_file, delimiter="\t")

        for row in reader:
            label = row[0].strip()
            img = row[3].strip().split("#")[0]
            labels.append(label2id[label])
            premise_tokens = row[1].strip().lower().split()
            hypothesis_tokens = row[2].strip().lower().split()
            premises.append([token2id.get(token, token2id["#unk#"]) for token in premise_tokens])
            hypotheses.append([token2id.get(token, token2id["#unk#"]) for token in hypothesis_tokens])
            img_names.append(img)

        premises = pad_sequences(premises, padding="post", value=token2id["#pad#"], dtype=np.long)
        hypotheses = pad_sequences(hypotheses, padding="post", value=token2id["#pad#"], dtype=np.long)
        labels = np.array(labels)

    return labels, premises, hypotheses, img_names


def load_ic_dataset(ic_dataset_filename, token2id, label2id):
    labels = []
    sentences = []
    img_names = []

    with open(ic_dataset_filename) as in_file:
        reader = csv.reader(in_file, delimiter="\t")

        for row in reader:
            label = row[0].strip()
            img = row[2].strip()
            labels.append(label2id[label])
            sentence_tokens = row[1].strip().lower().split()
            sentences.append([token2id.get(token, token2id["#unk#"]) for token in sentence_tokens])
            img_names.append(img)

        sentences = pad_sequences(sentences, padding="post", value=token2id["#pad#"], dtype=np.long)
        labels = np.array(labels)

    return labels, sentences, img_names


class ImageReader:
    def __init__(self, img_names_filename, img_features_filename):
        self._img_names_filename = img_names_filename
        self._img_features_filename = img_features_filename

        with open(img_names_filename) as in_file:
            img_names = json.load(in_file)

        with open(img_features_filename, mode="rb") as in_file:
            img_features = np.load(in_file)

        self._img_names_features = {filename: features for filename, features in zip(img_names, img_features)}

    def get_features(self, images_names):
        return np.array([self._img_names_features[image_name] for image_name in images_names])
