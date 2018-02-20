from argparse import ArgumentParser

import en_core_web_sm

from dataset import load_te_dataset
from embedding import load_glove

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--vectors_filename", type=str, required=True)
    parser.add_argument("--training_set_filename", type=str, required=True)
    args = parser.parse_args()
    embeddings, token2id, id2token = load_glove(args.vectors_filename, 300000, 300)
    print("girl" in token2id)
    print(embeddings[token2id["girl"]])
    label2id = {"neutral": 0, "entailment": 1, "contradiction": 2}
    id2label = {v: k for k, v in label2id.items()}
    spacy_nlp = en_core_web_sm.load()
    load_te_dataset(args.training_set_filename, token2id, label2id, spacy_nlp)
