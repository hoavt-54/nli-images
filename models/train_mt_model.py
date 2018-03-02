import atexit
import json
import os
import pickle
import random
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from datasets import load_ic_dataset, load_vte_dataset, ImageReader
from embeddings import load_glove
from tl_mt_model import build_tl_mt_model
from utils import start_logger, stop_logger

if __name__ == "__main__":
    random_seed = 12345
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
    parser = ArgumentParser()
    parser.add_argument("--ic_train_filename", type=str, required=True)
    parser.add_argument("--vte_train_filename", type=str, required=True)
    parser.add_argument("--ic_dev_filename", type=str, required=True)
    parser.add_argument("--vte_dev_filename", type=str, required=True)
    parser.add_argument("--vectors_filename", type=str, required=True)
    parser.add_argument("--ic_img_names_filename", type=str, required=True)
    parser.add_argument("--ic_img_features_filename", type=str, required=True)
    parser.add_argument("--vte_img_names_filename", type=str, required=True)
    parser.add_argument("--vte_img_features_filename", type=str, required=True)
    parser.add_argument("--model_save_filename", type=str, required=True)
    parser.add_argument("--max_vocab", type=int, default=300000)
    parser.add_argument("--embeddings_size", type=int, default=300)
    parser.add_argument("--train_embeddings", type=bool, default=True)
    parser.add_argument("--num_img_features", type=int, default=36)
    parser.add_argument("--img_features_size", type=int, default=2048)
    parser.add_argument("--rnn_hidden_size", type=int, default=512)
    parser.add_argument("--rnn_dropout_ratio", type=float, default=0.5)
    parser.add_argument("--multimodal_fusion_hidden_size", type=int, default=512)
    parser.add_argument("--classification_hidden_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    start_logger(args.model_save_filename + ".train_log")
    atexit.register(stop_logger)

    print("-- Building vocabulary")
    embeddings, token2id, id2token = load_glove(args.vectors_filename, args.max_vocab, args.embeddings_size)
    vte_label2id = {"neutral": 0, "entailment": 1, "contradiction": 2}
    vte_id2label = {v: k for k, v in vte_label2id.items()}
    ic_label2id = {"no": 0, "yes": 1}
    ic_id2label = {v: k for k, v in ic_label2id.items()}
    num_tokens = len(token2id)
    vte_num_labels = len(vte_label2id)
    ic_num_labels = len(ic_label2id)
    print("Number of tokens: {}".format(num_tokens))
    print("Number of VTE labels: {}".format(vte_num_labels))
    print("Number of IC labels: {}".format(ic_num_labels))

    with open(args.model_save_filename + ".params", mode="w") as out_file:
        json.dump(vars(args), out_file)
        print("Params saved to: {}".format(args.model_save_filename + ".params"))

        with open(args.model_save_filename + ".index", mode="wb") as out_file:
            pickle.dump(
                {
                    "token2id": token2id,
                    "id2token": id2token,
                    "vte_label2id": vte_label2id,
                    "vte_id2label": vte_id2label,
                    "ic_label2id": ic_label2id,
                    "ic_id2label": ic_id2label
                },
                out_file
            )
            print("Index saved to: {}".format(args.model_save_filename + ".index"))

    print("-- Loading training set")
    ic_train_labels, ic_train_sentences, ic_train_img_names, _ = load_ic_dataset(
        args.ic_train_filename,
        token2id,
        ic_label2id
    )

    print("-- Loading development set")
    ic_dev_labels, ic_dev_sentences, ic_dev_img_names, _ = load_ic_dataset(
        args.ic_dev_filename,
        token2id,
        ic_label2id
    )

    print("-- Loading training set")
    vte_train_labels, vte_train_premises, vte_train_hypotheses, vte_train_img_names, _, _ =\
        load_vte_dataset(
            args.vte_train_filename,
            token2id,
            vte_label2id
        )

    print("-- Loading development set")
    vte_dev_labels, vte_dev_premises, vte_dev_hypotheses, vte_dev_img_names, _, _ =\
        load_vte_dataset(
            args.vte_dev_filename,
            token2id,
            vte_label2id
        )

    print("-- Loading images")
    ic_image_reader = ImageReader(args.ic_img_names_filename, args.ic_img_features_filename)

    print("-- Loading images")
    vte_image_reader = ImageReader(args.vte_img_names_filename, args.vte_img_features_filename)

    sentence_input = tf.placeholder(tf.int32, (None, None), name="sentence_input")
    premise_input = tf.placeholder(tf.int32, (None, None), name="premise_input")
    hypothesis_input = tf.placeholder(tf.int32, (None, None), name="hypothesis_input")
    img_features_input = tf.placeholder(tf.float32, (None, args.num_img_features, args.img_features_size), name="img_features_input")
    ic_label_input = tf.placeholder(tf.int32, (None,), name="label_input")
    vte_label_input = tf.placeholder(tf.int32, (None,), name="vte_label_input")
    dropout_input = tf.placeholder(tf.float32, name="dropout_input")

    ic_logits, vte_logits = build_tl_mt_model(
        sentence_input,
        premise_input,
        hypothesis_input,
        img_features_input,
        dropout_input,
        num_tokens,
        ic_num_labels,
        vte_num_labels,
        None,
        args.embeddings_size,
        args.num_img_features,
        args.img_features_size,
        True,
        args.rnn_hidden_size,
        args.classification_hidden_size,
        args.multimodal_fusion_hidden_size
    )
