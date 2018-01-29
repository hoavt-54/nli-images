import os
import random
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from embedding import glove_embeddings_initializer


def build_top_down_baseline_model(premise_input,
                                  hypothesis_input,
                                  img_features_input,
                                  num_tokens,
                                  num_labels,
                                  embeddings,
                                  embeddings_size,
                                  num_img_features,
                                  train_embeddings,
                                  rnn_hidden_size):
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
    # premise_last = extract_axis_1(premise_outputs, premise_length - 1)
    hypothesis_outputs, hypothesis_final_states = tf.nn.dynamic_rnn(
        cell=lst_cell,
        inputs=hypothesis_embeddings,
        sequence_length=hypothesis_length,
        dtype=tf.float32
    )
    normalized_img_features = tf.nn.l2_normalize(img_features_input, dim=2)

    reshaped_premise = tf.reshape(tf.tile(premise_final_states.h, [1, num_img_features]), [8, 36, 100])
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

    reshaped_hypothesis = tf.reshape(tf.tile(hypothesis_final_states.h, [1, num_img_features]), [8, 36, 100])
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
                    activation_fn=tf.nn.tanh
                ),
                rnn_hidden_size * 2,
                activation_fn=tf.nn.tanh
            ),
            rnn_hidden_size * 2,
            activation_fn=tf.nn.tanh
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
    parser.add_argument("--embeddings_size", type=int, default=300)
    parser.add_argument("--train_embeddings", type=bool, default=True)
    parser.add_argument("--num_img_features", type=int, default=36)
    parser.add_argument("--img_features_size", type=int, default=2048)
    parser.add_argument("--rnn_hidden_size", type=int, default=100)
    num_tokens = 10
    num_labels = 3
    embeddings = None
    batch_size = 8
    max_premise_length = 12
    max_hypothesis_length = 10
    args = parser.parse_args()

    premise_input = tf.placeholder(tf.int32, (None, None), name="premise_input")
    hypothesis_input = tf.placeholder(tf.int32, (None, None), name="hypothesis_input")
    img_features_input = tf.placeholder(tf.float32, (None, args.num_img_features, args.img_features_size),
                                        name="img_features_input")
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
    )
    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1)) as session:
        session.run(tf.global_variables_initializer())
        results = session.run(logits, feed_dict={
            premise_input: np.random.randint(1, num_tokens, (batch_size, max_premise_length)),
            hypothesis_input: np.random.randint(1, num_tokens, (batch_size, max_hypothesis_length)),
            img_features_input: np.random.randn(batch_size, args.num_img_features, args.img_features_size),
        })
        print(results)
