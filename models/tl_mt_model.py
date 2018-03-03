from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper

from embeddings import glove_embeddings_initializer
from utils import gated_tanh
import tensorflow as tf

import numpy as np


def build_tl_mt_model(sentence_input,
                      premise_input,
                      hypothesis_input,
                      img_features_input,
                      dropout_input,
                      num_tokens,
                      num_ic_labels,
                      num_vte_labels,
                      embeddings,
                      embeddings_size,
                      num_img_features,
                      img_features_size,
                      train_embeddings,
                      rnn_hidden_size,
                      multimodal_fusion_hidden_size,
                      classification_hidden_size):
    sentence_length = tf.cast(
        tf.reduce_sum(
            tf.cast(tf.not_equal(sentence_input, tf.zeros_like(sentence_input, dtype=tf.int32)), tf.int64),
            1
        ),
        tf.int32
    )
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
    sentence_embeddings = tf.nn.embedding_lookup(embedding_matrix, sentence_input)
    premise_embeddings = tf.nn.embedding_lookup(embedding_matrix, premise_input)
    hypothesis_embeddings = tf.nn.embedding_lookup(embedding_matrix, hypothesis_input)
    lstm_cell = DropoutWrapper(
        tf.nn.rnn_cell.LSTMCell(rnn_hidden_size),
        input_keep_prob=dropout_input,
        output_keep_prob=dropout_input
    )
    sentence_outputs, sentence_final_states = tf.nn.dynamic_rnn(
        cell=lstm_cell,
        inputs=sentence_embeddings,
        sequence_length=sentence_length,
        dtype=tf.float32
    )
    premise_outputs, premise_final_states = tf.nn.dynamic_rnn(
        cell=lstm_cell,
        inputs=premise_embeddings,
        sequence_length=premise_length,
        dtype=tf.float32
    )
    hypothesis_outputs, hypothesis_final_states = tf.nn.dynamic_rnn(
        cell=lstm_cell,
        inputs=hypothesis_embeddings,
        sequence_length=hypothesis_length,
        dtype=tf.float32
    )
    normalized_img_features = tf.nn.l2_normalize(img_features_input, dim=2)

    reshaped_sentence = tf.reshape(tf.tile(sentence_final_states.h, [1, num_img_features]), [-1, num_img_features, rnn_hidden_size])
    img_sentence_concatenation = tf.concat([normalized_img_features, reshaped_sentence], -1)
    gated_img_sentence_concatenation = tf.nn.dropout(
        gated_tanh(img_sentence_concatenation, rnn_hidden_size),
        keep_prob=dropout_input
    )
    att_wa_sentence = lambda x: tf.nn.dropout(
        tf.contrib.layers.fully_connected(x, 1, activation_fn=None, biases_initializer=None),
        keep_prob=dropout_input
    )
    a_sentence = att_wa_sentence(gated_img_sentence_concatenation)
    a_sentence = tf.nn.softmax(tf.squeeze(a_sentence))
    v_head_sentence = tf.squeeze(tf.matmul(tf.expand_dims(a_sentence, 1), normalized_img_features))

    gated_sentence = tf.nn.dropout(
        gated_tanh(sentence_final_states.h, multimodal_fusion_hidden_size),
        keep_prob=dropout_input
    )

    v_head_sentence.set_shape((premise_embeddings.get_shape()[0], img_features_size))
    gated_img_features_sentence = tf.nn.dropout(
        gated_tanh(v_head_sentence, multimodal_fusion_hidden_size),
        keep_prob=dropout_input
    )

    h_premise_img = tf.multiply(gated_sentence, gated_img_features_sentence)
    gated_first_layer = tf.nn.dropout(
        gated_tanh(h_premise_img, classification_hidden_size),
        keep_prob=dropout_input
    )
    gated_second_layer = tf.nn.dropout(
        gated_tanh(gated_first_layer, classification_hidden_size),
        keep_prob=dropout_input
    )
    gated_third_layer = tf.nn.dropout(
        gated_tanh(gated_second_layer, classification_hidden_size),
        keep_prob=dropout_input
    )

    ic_classification = tf.nn.dropout(
        tf.contrib.layers.fully_connected(
            gated_third_layer,
            num_ic_labels,
            activation_fn=None
        ),
        keep_prob=dropout_input
    )

    reshaped_premise = tf.reshape(tf.tile(premise_final_states.h, [1, num_img_features]), [-1, num_img_features, rnn_hidden_size])
    img_premise_concatenation = tf.concat([normalized_img_features, reshaped_premise], -1)
    gated_img_premise_concatenation = tf.nn.dropout(
        gated_tanh(img_premise_concatenation, rnn_hidden_size),
        keep_prob=dropout_input
    )
    att_wa_premise = lambda x: tf.nn.dropout(
        tf.contrib.layers.fully_connected(x, 1, activation_fn=None, biases_initializer=None),
        keep_prob=dropout_input
    )
    a_premise = att_wa_premise(gated_img_premise_concatenation)
    a_premise = tf.nn.softmax(tf.squeeze(a_premise))
    v_head_premise = tf.squeeze(tf.matmul(tf.expand_dims(a_premise, 1), normalized_img_features))

    reshaped_hypothesis = tf.reshape(tf.tile(hypothesis_final_states.h, [1, num_img_features]), [-1, num_img_features, rnn_hidden_size])
    img_hypothesis_concatenation = tf.concat([normalized_img_features, reshaped_hypothesis], -1)
    gated_img_hypothesis_concatenation = tf.nn.dropout(
        gated_tanh(img_hypothesis_concatenation, rnn_hidden_size),
        keep_prob=dropout_input
    )
    att_wa_hypothesis = lambda x: tf.nn.dropout(
        tf.contrib.layers.fully_connected(x, 1, activation_fn=None, biases_initializer=None),
        keep_prob=dropout_input
    )
    a_hypothesis = att_wa_hypothesis(gated_img_hypothesis_concatenation)
    a_hypothesis = tf.nn.softmax(tf.squeeze(a_hypothesis))
    v_head_hypothesis = tf.squeeze(tf.matmul(tf.expand_dims(a_hypothesis, 1), normalized_img_features))

    gated_premise = tf.nn.dropout(
        gated_tanh(premise_final_states.h, multimodal_fusion_hidden_size),
        keep_prob=dropout_input
    )
    gated_hypothesis = tf.nn.dropout(
        gated_tanh(hypothesis_final_states.h, multimodal_fusion_hidden_size),
        keep_prob=dropout_input
    )

    v_head_premise.set_shape((premise_embeddings.get_shape()[0], img_features_size))
    gated_img_features_premise = tf.nn.dropout(
        gated_tanh(v_head_premise, multimodal_fusion_hidden_size),
        keep_prob=dropout_input
    )

    v_head_hypothesis.set_shape((hypothesis_embeddings.get_shape()[0], img_features_size))
    gated_img_features_hypothesis = tf.nn.dropout(
        gated_tanh(v_head_hypothesis, multimodal_fusion_hidden_size),
        keep_prob=dropout_input
    )

    h_premise_img = tf.multiply(gated_premise, gated_img_features_premise)
    h_hypothesis_img = tf.multiply(gated_hypothesis, gated_img_features_hypothesis)
    final_concatenation = tf.concat([h_premise_img, h_hypothesis_img], 1)
    gated_first_layer = tf.nn.dropout(
        gated_tanh(final_concatenation, classification_hidden_size),
        keep_prob=dropout_input
    )
    gated_second_layer = tf.nn.dropout(
        gated_tanh(gated_first_layer, classification_hidden_size),
        keep_prob=dropout_input
    )
    gated_third_layer = tf.nn.dropout(
        gated_tanh(gated_second_layer, classification_hidden_size),
        keep_prob=dropout_input
    )

    vte_classification = tf.nn.dropout(
        tf.contrib.layers.fully_connected(
            gated_third_layer,
            num_vte_labels,
            activation_fn=None
        ),
        keep_prob=dropout_input
    )

    return ic_classification, vte_classification


if __name__ == "__main__":
    batch_size = 32
    max_sentence_length = 15
    max_premise_length = 25
    max_hypothesis_length = 35
    num_tokens = 20
    num_ic_labels = 2
    num_vte_labels = 3
    embeddings_size = 50
    num_img_features = 32
    img_features_size = 2048
    rnn_hidden_size = 100
    classification_hidden_size = 100
    multimodal_fusion_hidden_size = 100
    sentence_input = tf.placeholder(tf.int32, (None, None), name="sentence_input")
    premise_input = tf.placeholder(tf.int32, (None, None), name="premise_input")
    hypothesis_input = tf.placeholder(tf.int32, (None, None), name="hypothesis_input")
    img_features_input = tf.placeholder(tf.float32, (None, num_img_features, img_features_size),
                                        name="img_features_input")
    ic_label_input = tf.placeholder(tf.int32, (None,), name="ic_label_input")
    vte_label_input = tf.placeholder(tf.int32, (None,), name="vte_label_input")
    dropout_input = tf.placeholder(tf.float32, name="dropout_input")

    ic_logits, vte_logits = build_tl_mt_model(
        sentence_input,
        premise_input,
        hypothesis_input,
        img_features_input,
        dropout_input,
        num_tokens,
        num_ic_labels,
        num_vte_labels,
        None,
        embeddings_size,
        num_img_features,
        img_features_size,
        True,
        rnn_hidden_size,
        classification_hidden_size,
        multimodal_fusion_hidden_size
    )

    ic_loss_function = tf.losses.sparse_softmax_cross_entropy(ic_label_input, ic_logits)
    ic_train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(ic_loss_function)

    vte_loss_function = tf.losses.sparse_softmax_cross_entropy(vte_label_input, vte_logits)
    vte_train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(vte_loss_function)

    sentences = np.random.randint(1, 10, (batch_size, max_sentence_length))
    premises = np.random.randint(5, 20, (batch_size, max_premise_length))
    hypotheses = np.random.randint(5, 20, (batch_size, max_hypothesis_length))
    ic_labels = np.random.randint(0, 2, (batch_size))
    vte_labels = np.random.randint(0, 3, (batch_size))
    img_features = np.random.randn(batch_size, num_img_features, img_features_size)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())

        # Forward propagations

        ic_result = session.run(ic_logits, feed_dict={
            sentence_input: sentences,
            img_features_input: img_features,
            dropout_input: 1.0
        })
        print(ic_result)

        vte_result = session.run(vte_logits, feed_dict={
            premise_input: premises,
            hypothesis_input: hypotheses,
            img_features_input: img_features,
            dropout_input: 1.0
        })
        print(vte_result)

        # Backward propagations

        _, ic_loss_value = session.run([ic_train_step, ic_loss_function], feed_dict={
            sentence_input: sentences,
            img_features_input: img_features,
            ic_label_input: ic_labels,
            dropout_input: 1.0
        })
        print("ic_loss_value", ic_loss_value)

        _, vte_loss_value = session.run([vte_train_step, vte_loss_function], feed_dict={
            premise_input: premises,
            hypothesis_input: hypotheses,
            img_features_input: img_features,
            vte_label_input: vte_labels,
            dropout_input: 1.0
        })
        print("vte_loss_value", vte_loss_value)

        _, ic_loss_value = session.run([ic_train_step, ic_loss_function], feed_dict={
            sentence_input: sentences,
            img_features_input: img_features,
            ic_label_input: ic_labels,
            dropout_input: 1.0
        })
        print("ic_loss_value", ic_loss_value)
