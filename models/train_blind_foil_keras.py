import csv
from argparse import ArgumentParser

import numpy as np
from keras import optimizers
from keras.layers import Dense, Masking
from keras.layers import Embedding
from keras.layers import LSTM
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


def load_glove(filename, max_vocab):
    token2id = {}
    id2token = {}

    token_id = len(token2id)
    token2id["#pad#"] = token_id
    id2token[token_id] = "#pad#"

    token_id = len(token2id)
    token2id["#unk#"] = token_id
    id2token[token_id] = "#unk#"

    with open(filename) as in_file:
        for line_index, line in enumerate(in_file):
            values = line.rstrip().split(" ")
            word = values[0]
            token_id = len(token2id)
            token2id[word] = token_id
            id2token[token_id] = word

            if token_id == max_vocab + 1:
                break

    return token2id, id2token


def load_dataset(ic_dataset_filename, token2id, label2id):
    padded_sentences = []
    labels = []

    with open(ic_dataset_filename) as in_file:
        reader = csv.reader(in_file, delimiter="\t")

        for row in reader:
            label = row[0].strip()
            sentence_tokens = row[1].strip().lower().split()
            labels.append(label2id[label])
            padded_sentences.append([token2id.get(token, token2id["#unk#"]) for token in sentence_tokens])

        padded_sentences = pad_sequences(padded_sentences, padding="post", value=token2id["#pad#"], dtype=np.long)
        labels = np.array(labels)

    return labels, padded_sentences


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_filename", type=str, required=True)
    parser.add_argument("--test_filename", type=str, required=True)
    parser.add_argument("--vectors_filename", type=str, required=True)
    parser.add_argument("--max_vocab", type=int, default=300000)
    parser.add_argument("--embeddings_size", type=int, default=300)
    parser.add_argument("--rnn_hidden_size", type=int, default=512)
    parser.add_argument("--classification_hidden_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()

    token2id, id2token = load_glove(args.vectors_filename, args.max_vocab)
    label2id = {"no": 0, "yes": 1}
    id2label = {v: k for k, v in label2id.items()}
    num_tokens = len(token2id)
    num_labels = len(label2id)
    print("Number of tokens: {}".format(num_tokens))
    print("Number of labels: {}".format(num_labels))

    print("-- Loading training set")
    train_labels, train_sentences = load_dataset(args.train_filename, token2id, label2id)

    print("-- Loading test set")
    test_labels, test_sentences = load_dataset(args.test_filename, token2id, label2id)

    model = Sequential()
    model.add(Embedding(num_tokens, output_dim=args.embeddings_size, mask_zero=True))
    model.add(Masking(token2id["#pad#"]))
    model.add(LSTM(args.rnn_hidden_size))
    model.add(Dense(args.classification_hidden_size, activation="relu"))
    model.add(Dense(num_labels))

    def loss_fn(y_true, y_pred):
        y_true = tf.squeeze(y_true)
        y_true = tf.cast(y_true, tf.int32)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred)

    opt = optimizers.adam(lr=args.learning_rate , epsilon=1e-08)
    model.compile(loss=loss_fn, optimizer=opt)
    model.fit(train_sentences, train_labels, batch_size=args.batch_size, epochs=args.num_epochs)

    predictions = np.argmax(model.predict(test_sentences, batch_size=args.batch_size, verbose=0), axis=1)
    print((predictions == test_labels).sum() / test_labels.shape[0])
