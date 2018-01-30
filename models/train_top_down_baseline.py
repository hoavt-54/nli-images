import json
import os
import pickle
import random
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from dataset import ImageReader, load_vte_dataset
from embedding import glove_embeddings_initializer, load_glove
from progress import Progbar
from utils import batch


def build_top_down_baseline_model(premise_input,
                                  hypothesis_input,
                                  img_features_input,
                                  num_tokens,
                                  num_labels,
                                  embeddings,
                                  embeddings_size,
                                  num_img_features,
                                  train_embeddings,
                                  rnn_hidden_size,
                                  batch_size):
    def _gated_tanh(x, W, W_prime):
        y_tilde = tf.nn.tanh(W(x))
        g = tf.nn.sigmoid(W_prime(x))
        return tf.multiply(y_tilde, g)

    premise_length = tf.cast(
        tf.reduce_sum(
            tf.cast(tf.not_equal(premise_input, tf.zeros_like(premise_input, dtype=tf.int32)), tf.int64),
            1
        ),
        tf.int32
    )
    hypothesis_length = tf.cast(
        tf.reduce_sum(
            tf.cast(tf.not_equal(hypothesis_input, tf.zeros_like(hypothesis_input, dtype=tf.int32)), tf.int64),
            1
        ),
        tf.int32
    )
    if embeddings is not None:
        embedding_matrix = tf.get_variable(
            "embedding_matrix",
            shape=(num_tokens, embeddings_size),
            initializer=glove_embeddings_initializer(embeddings),
            trainable=train_embeddings
        )
        print("Loaded GloVe embeddings!")
    else:
        embedding_matrix = tf.get_variable(
            "embedding_matrix",
            shape=(num_tokens, embeddings_size),
            initializer=tf.random_normal_initializer(stddev=0.05),
            trainable=train_embeddings
        )
    premise_embeddings = tf.nn.embedding_lookup(embedding_matrix, premise_input)
    hypothesis_embeddings = tf.nn.embedding_lookup(embedding_matrix, hypothesis_input)
    lst_cell = tf.nn.rnn_cell.LSTMCell(rnn_hidden_size)
    premise_outputs, premise_final_states = tf.nn.dynamic_rnn(
        cell=lst_cell,
        inputs=premise_embeddings,
        sequence_length=premise_length,
        dtype=tf.float32
    )
    hypothesis_outputs, hypothesis_final_states = tf.nn.dynamic_rnn(
        cell=lst_cell,
        inputs=hypothesis_embeddings,
        sequence_length=hypothesis_length,
        dtype=tf.float32
    )

    normalized_img_features = tf.nn.l2_normalize(img_features_input, dim=2)

    reshaped_premise = tf.reshape(tf.tile(premise_final_states.h, [1, num_img_features]), [batch_size, num_img_features, rnn_hidden_size])
    img_premise_concatenation = tf.concat([normalized_img_features, reshaped_premise], -1)
    gated_W_premise_img_att = lambda x: tf.contrib.layers.fully_connected(x, rnn_hidden_size)
    gated_W_prime_premise_img_att = lambda x: tf.contrib.layers.fully_connected(x, rnn_hidden_size)
    gated_img_premise_concatenation = _gated_tanh(
        img_premise_concatenation,
        gated_W_premise_img_att,
        gated_W_prime_premise_img_att
    )
    att_wa_premise = lambda x: tf.contrib.layers.fully_connected(x, 1)
    a_premise = att_wa_premise(gated_img_premise_concatenation)
    a_premise = tf.nn.softmax(tf.squeeze(a_premise))
    v_head_premise = tf.squeeze(tf.matmul(tf.expand_dims(a_premise, 1), normalized_img_features))

    reshaped_hypothesis = tf.reshape(tf.tile(hypothesis_final_states.h, [1, num_img_features]), [batch_size, num_img_features, rnn_hidden_size])
    img_hypothesis_concatenation = tf.concat([normalized_img_features, reshaped_hypothesis], -1)
    gated_W_hypothesis_img_att = lambda x: tf.contrib.layers.fully_connected(x, rnn_hidden_size)
    gated_W_prime_hypothesis_img_att = lambda x: tf.contrib.layers.fully_connected(x, rnn_hidden_size)
    gated_img_hypothesis_concatenation = _gated_tanh(
        img_hypothesis_concatenation,
        gated_W_hypothesis_img_att,
        gated_W_prime_hypothesis_img_att
    )
    att_wa_hypothesis = lambda x: tf.contrib.layers.fully_connected(x, 1)
    a_hypothesis = att_wa_hypothesis(gated_img_hypothesis_concatenation)
    a_hypothesis = tf.nn.softmax(tf.squeeze(a_hypothesis))
    v_head_hypothesis = tf.squeeze(tf.matmul(tf.expand_dims(a_hypothesis, 1), normalized_img_features))

    gated_W_premise = lambda x: tf.contrib.layers.fully_connected(x, rnn_hidden_size)
    gated_W_prime_premise = lambda x: tf.contrib.layers.fully_connected(x, rnn_hidden_size)
    gated_premise = _gated_tanh(premise_final_states.h, gated_W_premise, gated_W_prime_premise)

    gated_W_hypothesis = lambda x: tf.contrib.layers.fully_connected(x, rnn_hidden_size)
    gated_W_prime_hypothesis = lambda x: tf.contrib.layers.fully_connected(x, rnn_hidden_size)
    gated_hypothesis = _gated_tanh(premise_final_states.h, gated_W_hypothesis, gated_W_prime_hypothesis)

    gated_W_img_premise = lambda x: tf.contrib.layers.fully_connected(x, rnn_hidden_size)
    gated_W_prime_img_premise = lambda x: tf.contrib.layers.fully_connected(x, rnn_hidden_size)
    gated_img_features_premise = _gated_tanh(v_head_premise, gated_W_img_premise, gated_W_prime_img_premise)

    gated_W_img_hypothesis = lambda x: tf.contrib.layers.fully_connected(x, rnn_hidden_size)
    gated_W_prime_img_hypothesis = lambda x: tf.contrib.layers.fully_connected(x, rnn_hidden_size)
    gated_img_features_hypothesis = _gated_tanh(v_head_hypothesis, gated_W_img_hypothesis, gated_W_prime_img_hypothesis)

    h_premise_img = tf.multiply(gated_premise, gated_img_features_premise)
    h_hypothesis_img = tf.multiply(gated_hypothesis, gated_img_features_hypothesis)
    h = tf.concat([h_premise_img, h_hypothesis_img], 1)

    return tf.contrib.layers.fully_connected(
        tf.contrib.layers.fully_connected(
            tf.contrib.layers.fully_connected(
                tf.contrib.layers.fully_connected(
                    h,
                    rnn_hidden_size * 2,
                    activation_fn=tf.nn.relu
                ),
                rnn_hidden_size * 2,
                activation_fn=tf.nn.relu
            ),
            rnn_hidden_size * 2,
            activation_fn=tf.nn.relu
        ),
        num_labels,
        activation_fn=None
    )


if __name__ == "__main__":
    random_seed = 12345
    os.environ["PYTHONHASHSEED"] = str(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
    parser = ArgumentParser()
    parser.add_argument("--train_filename", type=str, required=True)
    parser.add_argument("--vectors_filename", type=str, required=True)
    parser.add_argument("--img_names_filename", type=str, required=True)
    parser.add_argument("--img_features_filename", type=str, required=True)
    parser.add_argument("--model_save_filename", type=str, required=True)
    parser.add_argument("--max_vocab", type=int, default=300000)
    parser.add_argument("--embeddings_size", type=int, default=300)
    parser.add_argument("--train_embeddings", type=bool, default=True)
    parser.add_argument("--num_img_features", type=int, default=36)
    parser.add_argument("--img_features_size", type=int, default=2048)
    parser.add_argument("--rnn_hidden_size", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1.0)
    args = parser.parse_args()

    print("-- Building vocabulary")
    embeddings, token2id, id2token = load_glove(args.vectors_filename, args.max_vocab, args.embeddings_size)
    label2id = {"neutral": 0, "entailment": 1, "contradiction": 2}
    id2label = {v: k for k, v in label2id.items()}
    num_tokens = len(token2id)
    num_labels = len(label2id)
    print("Number of tokens: {}".format(num_tokens))
    print("Number of labels: {}".format(num_labels))

    with open(args.model_save_filename + ".params", mode="w") as out_file:
        json.dump(vars(args), out_file)
        print("Params saved to: {}".format(args.model_save_filename + ".params"))

        with open(args.model_save_filename + ".index", mode="wb") as out_file:
            pickle.dump(
                {
                    "token2id": token2id,
                    "id2token": id2token,
                    "label2id": label2id,
                    "id2label": id2label
                },
                out_file
            )
            print("Index saved to: {}".format(args.model_save_filename + ".index"))

    print("-- Loading training set")
    train_labels, train_premises, train_hypotheses, train_img_names = load_vte_dataset(args.train_filename, token2id, label2id)

    print("-- Loading development set")
    dev_labels, dev_premises, dev_hypotheses, dev_img_names = load_vte_dataset(args.dev_filename, token2id, label2id)

    print("-- Loading images")
    image_reader = ImageReader(args.img_names_filename, args.img_features_filename)

    print("-- Building model")
    premise_input = tf.placeholder(tf.int32, (None, None), name="premise_input")
    hypothesis_input = tf.placeholder(tf.int32, (None, None), name="hypothesis_input")
    img_features_input = tf.placeholder(tf.float32, (None, args.num_img_features, args.img_features_size), name="img_features_input")
    label_input = tf.placeholder(tf.int32, (None,), name="label_input")
    logits = build_top_down_baseline_model(
        premise_input,
        hypothesis_input,
        img_features_input,
        num_tokens,
        num_labels,
        embeddings,
        args.embeddings_size,
        args.num_img_features,
        args.train_embeddings,
        args.rnn_hidden_size,
        args.batch_size
    )
    loss_function = tf.losses.sparse_softmax_cross_entropy(label_input, logits)
    train_step = tf.train.AdadeltaOptimizer(learning_rate=args.learning_rate).minimize(loss_function)

    num_examples = train_labels.shape[0]
    num_batches = num_examples // args.batch_size

    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1)) as session:
        session.run(tf.global_variables_initializer())

        for epoch in range(args.num_epochs):
            print("\n==> Online epoch # {0}".format(epoch + 1))
            progress = Progbar(num_batches)
            batches_indexes = np.arange(num_examples)
            np.random.shuffle(batches_indexes)
            batch_index = 1
            epoch_loss = 0

            for indexes in batch(batches_indexes, args.batch_size):
                batch_premises = train_premises[indexes]
                batch_hypotheses = train_hypotheses[indexes]
                batch_labels = train_labels[indexes]
                batch_img_names = [train_img_names[i] for i in indexes]
                batch_img_features = image_reader.get_features(batch_img_names)

                loss, _ = session.run([loss_function, train_step], feed_dict={
                    premise_input: batch_premises,
                    hypothesis_input: batch_hypotheses,
                    img_features_input: batch_img_features,
                    label_input: batch_labels
                })
                progress.update(batch_index, [("Loss", loss)])
                epoch_loss += loss
                batch_index += 1
            print("Current mean training loss: {}\n".format(epoch_loss / num_batches))

            print("-- Validating model")
            dev_num_examples = dev_labels.shape[0]
            dev_batches_indexes = np.arange(dev_num_examples)
            dev_num_correct = 0

            for indexes in batch(dev_batches_indexes, args.batch_size):
                dev_batch_premises = dev_premises[indexes]
                dev_batch_hypotheses = dev_hypotheses[indexes]
                dev_batch_labels = dev_labels[indexes]
                dev_batch_img_names = [dev_img_names[i] for i in indexes]
                dev_batch_img_features = image_reader.get_features(dev_batch_img_names)
                predictions = session.run(
                    tf.argmax(logits, axis=1),
                    feed_dict={
                        premise_input: dev_batch_premises,
                        hypothesis_input: dev_batch_hypotheses,
                        img_features_input: dev_batch_img_features
                    }
                )
                dev_num_correct += (predictions == dev_batch_labels).sum()
            dev_accuracy = dev_num_correct / dev_num_examples
            print("Current mean validation accuracy: {}".format(dev_accuracy))
