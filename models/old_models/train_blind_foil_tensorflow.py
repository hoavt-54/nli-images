import csv
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import Progbar


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


def build_vocabulary(ic_train_filename, ic_test_filename):
    token2id = {}
    id2token = {}

    token_id = len(token2id)
    token2id["#pad#"] = token_id
    id2token[token_id] = "#pad#"

    with open(ic_train_filename) as in_file:
        reader = csv.reader(in_file, delimiter="\t")

        for row in reader:
            sentence_tokens = row[1].strip().lower().split()

            for token in sentence_tokens:
                if token not in token2id:
                    token_id = len(token2id)
                    token2id[token] = token_id
                    id2token[token_id] = token

    with open(ic_test_filename) as in_file:
        reader = csv.reader(in_file, delimiter="\t")
        for row in reader:
            sentence_tokens = row[1].strip().lower().split()

            for token in sentence_tokens:
                if token not in token2id:
                    token_id = len(token2id)
                    token2id[token] = token_id
                    id2token[token_id] = token

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
            padded_sentences.append([token2id[token] for token in sentence_tokens])
            # padded_sentences.append([token2id.get(token, token2id["#unk#"]) for token in sentence_tokens])

        padded_sentences = pad_sequences(padded_sentences, padding="post", value=token2id["#pad#"], dtype=np.long)
        labels = np.array(labels)

    return labels, padded_sentences


def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_filename", type=str, required=True)
    parser.add_argument("--test_filename", type=str, required=True)
    parser.add_argument("--vectors_filename", type=str, required=False)
    parser.add_argument("--max_vocab", type=int, default=300000)
    parser.add_argument("--embeddings_size", type=int, default=300)
    parser.add_argument("--rnn_hidden_size", type=int, default=512)
    parser.add_argument("--classification_hidden_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    args = parser.parse_args()

    # token2id, id2token = load_glove(args.vectors_filename, args.max_vocab)
    token2id, id2token = build_vocabulary(args.train_filename, args.test_filename)
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

    sentence_input = tf.placeholder(tf.int32, (None, None), name="sentence_input")
    label_input = tf.placeholder(tf.int32, (None,), name="label_input")
    sentence_length = tf.cast(
        tf.reduce_sum(
            tf.cast(tf.not_equal(sentence_input, tf.zeros_like(sentence_input, dtype=tf.int32)), tf.int64),
            1
        ),
        tf.int32
    )
    embedding_matrix = tf.get_variable(
        "embedding_matrix",
        shape=(num_tokens, args.embeddings_size),
        initializer=tf.random_uniform_initializer()
    )
    sentence_embeddings = tf.nn.embedding_lookup(embedding_matrix, sentence_input)
    lstm_cell = tf.nn.rnn_cell.LSTMCell(args.rnn_hidden_size)
    sentence_outputs, sentence_final_states = tf.nn.dynamic_rnn(
        cell=lstm_cell,
        inputs=sentence_embeddings,
        sequence_length=sentence_length,
        dtype=tf.float32
    )
    logits = tf.contrib.layers.fully_connected(
        tf.contrib.layers.fully_connected(
            sentence_final_states.h,
            args.classification_hidden_size,
            activation_fn=tf.nn.relu
        ),
        num_labels,
        activation_fn=None
    )

    loss_function = tf.losses.sparse_softmax_cross_entropy(label_input, logits)
    train_step = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss_function)

    num_examples = train_labels.shape[0]
    num_batches = num_examples // args.batch_size

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        for epoch in range(args.num_epochs):
            print("\n==> Online epoch # {0}".format(epoch + 1))

            progress = Progbar(num_batches)
            batches_indexes = np.arange(num_examples)
            np.random.shuffle(batches_indexes)
            batch_index = 1

            for indexes in batch(batches_indexes, args.batch_size):
                batch_sentences = train_sentences[indexes]
                batch_labels = train_labels[indexes]

                loss, _ = session.run([loss_function, train_step], feed_dict={
                    sentence_input: batch_sentences,
                    label_input: batch_labels
                })
                progress.update(batch_index, [("Loss", loss)])
                batch_index += 1

            print("-- Evaluating model")

            test_num_examples = test_labels.shape[0]
            test_batches_indexes = np.arange(test_num_examples)
            test_num_correct = 0

            for indexes in batch(test_batches_indexes, args.batch_size):
                test_batch_sentences = test_sentences[indexes]
                test_batch_labels = test_labels[indexes]
                predictions = session.run(
                    tf.argmax(logits, axis=1),
                    feed_dict={
                        sentence_input: test_batch_sentences
                    }
                )
                test_num_correct += (predictions == test_batch_labels).sum()
                # print("---")
                # print("test_batch_sentences[0]", test_batch_sentences[0])
                # print("test_batch_sentences[1]", test_batch_sentences[1])
                # print("test_batch_sentences[2]", test_batch_sentences[2])
                # print("predictions", predictions)
                # print("test_batch_labels", test_batch_labels)
                # print("(predictions == test_batch_labels).sum()", (predictions == test_batch_labels).sum())
                # print("---")
            test_accuracy = test_num_correct / test_num_examples
            print("Current mean test accuracy: {}".format(test_accuracy))
