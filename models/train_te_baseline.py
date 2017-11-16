import atexit
import json
import pickle
import random
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import DropoutWrapper

from dataset import load_te_dataset
from embedding import load_glove, glove_embeddings_initializer
from logger import start_logger, stop_logger
from progress import Progbar
from utils import AttrDict


def build_te_baseline_model(premise_input,
                            hypothesis_input,
                            dropout_input,
                            num_tokens,
                            num_labels,
                            embeddings,
                            embeddings_size,
                            train_embeddings,
                            rnn_hidden_size):
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
    lst_cell = DropoutWrapper(
        tf.nn.rnn_cell.LSTMCell(rnn_hidden_size),
        input_keep_prob=dropout_input,
        output_keep_prob=dropout_input
    )
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
    # hypothesis_last = extract_axis_1(hypothesis_outputs, hypothesis_length - 1)
    premise_hypothesis = tf.concat([premise_final_states.h, hypothesis_final_states.h], axis=1)
    return tf.contrib.layers.fully_connected(
        tf.contrib.layers.fully_connected(
            tf.contrib.layers.fully_connected(
                tf.contrib.layers.fully_connected(
                    premise_hypothesis,
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
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.set_random_seed(random_seed)
    parser = ArgumentParser()
    parser.add_argument("--train_filename", type=str, required=True)
    parser.add_argument("--dev_filename", type=str, required=True)
    parser.add_argument("--vectors_filename", type=str, required=True)
    parser.add_argument("--model_save_filename", type=str, required=True)
    parser.add_argument("--model_load_filename", type=str)
    parser.add_argument("--max_vocab", type=int, default=300000)
    parser.add_argument("--embeddings_size", type=int, default=300)
    parser.add_argument("--train_embeddings", type=bool, default=True)
    parser.add_argument("--rnn_hidden_size", type=int, default=100)
    parser.add_argument("--rnn_dropout_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1.0)
    parser.add_argument("--l2_reg", type=float, default=0.000005)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    start_logger(args.model_save_filename + ".train_log")
    atexit.register(stop_logger)

    if args.model_load_filename:
        print("-- Loading params")
        with open(args.model_load_filename + ".params", mode="r") as in_file:
            params = json.load(in_file)
            params["l2_reg"] = args.l2_reg
            params["train_filename"] = args.train_filename
            params["dev_filename"] = args.dev_filename
            params["model_save_filename"] = args.model_save_filename
            params["model_load_filename"] = args.model_load_filename
            args = AttrDict(params)
            with open(args.model_save_filename + ".params", mode="w") as out_file:
                json.dump(vars(args), out_file)
                print("Updated params saved to: {}".format(args.model_save_filename + ".params"))

        print("-- Loading index")
        with open(args.model_load_filename + ".index", mode="rb") as in_file:
            index = pickle.load(in_file)
            token2id = index["token2id"]
            id2token = index["id2token"]
            label2id = index["label2id"]
            id2label = index["id2label"]
            num_tokens = len(token2id)
            num_labels = len(label2id)
    else:
        with open(args.model_save_filename + ".params", mode="w") as out_file:
            json.dump(vars(args), out_file)
            print("Params saved to: {}".format(args.model_save_filename + ".params"))

        print("-- Building vocabulary")
        embeddings, token2id, id2token = load_glove(args.vectors_filename, args.max_vocab, args.embeddings_size)
        label2id = {"neutral": 0, "entailment": 1, "contradiction": 2}
        id2label = {v: k for k, v in label2id.items()}
        num_tokens = len(token2id)
        num_labels = len(label2id)
        print("Number of tokens: {}".format(num_tokens))
        print("Number of labels: {}".format(num_labels))
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
    train_labels, train_premises, train_hypotheses = load_te_dataset(args.train_filename, token2id, label2id)

    print("-- Loading development set")
    dev_labels, dev_premises, dev_hypotheses = load_te_dataset(args.dev_filename, token2id, label2id)

    if not args.model_load_filename:
        print("-- Building model")
        premise_input = tf.placeholder(tf.int32, (None, None), name="premise_input")
        hypothesis_input = tf.placeholder(tf.int32, (None, None), name="hypothesis_input")
        label_input = tf.placeholder(tf.int32, (None,), name="label_input")
        dropout_input = tf.placeholder(tf.float32, name="dropout_input")
        logits = build_te_baseline_model(
            premise_input,
            hypothesis_input,
            dropout_input,
            num_tokens,
            num_labels,
            embeddings,
            args.embeddings_size,
            args.train_embeddings,
            args.rnn_hidden_size
        )
        L2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name]) * args.l2_reg
        loss_function = tf.losses.sparse_softmax_cross_entropy(label_input, logits) + L2_loss
        train_step = tf.train.AdadeltaOptimizer(learning_rate=args.learning_rate).minimize(loss_function)
        saver = tf.train.Saver()
        tf.add_to_collection("premise_input", premise_input)
        tf.add_to_collection("hypothesis_input", hypothesis_input)
        tf.add_to_collection("label_input", label_input)
        tf.add_to_collection("dropout_input", dropout_input)
        tf.add_to_collection("loss_function", loss_function)
        tf.add_to_collection("train_step", train_step)
        tf.add_to_collection("logits", logits)
    else:
        print("-- Loading model")
        saver = tf.train.import_meta_graph(args.model_load_filename + ".ckpt.meta")

    num_examples = train_labels.shape[0]
    num_batches = num_examples // args.batch_size
    dev_best_accuracy = -1
    stopping_step = 0
    best_epoch = None
    should_stop = False

    with tf.Session() as session:
        if not args.model_load_filename:
            session.run(tf.global_variables_initializer())
        else:
            saver.restore(session, args.model_load_filename + ".ckpt")
            train_step = tf.get_collection("train_step")[0]
            loss_function = tf.get_collection("loss_function")[0]
            dropout_input = tf.get_collection("dropout_input")[0]
            label_input = tf.get_collection("label_input")[0]
            hypothesis_input = tf.get_collection("hypothesis_input")[0]
            premise_input = tf.get_collection("premise_input")[0]
            logits = tf.get_collection("logits")[0]
            softmax_layer_weights = [v for v in tf.global_variables() if v.name == "fully_connected_3/weights:0"][0]
            session.run(softmax_layer_weights.initializer)

        for epoch in range(args.num_epochs):
            if should_stop:
                break

            print("\n==> Online epoch # {0}".format(epoch + 1))
            progress = Progbar(num_batches)
            batches_indexes = np.arange(num_examples)
            np.random.shuffle(batches_indexes)
            batch_index = 1
            epoch_loss = 0

            for start_idx in range(0, num_examples - args.batch_size + 1, args.batch_size):
                batch_indexes = batches_indexes[start_idx:start_idx + args.batch_size]
                batch_premises = train_premises[batch_indexes]
                batch_hypotheses = train_hypotheses[batch_indexes]
                batch_labels = train_labels[batch_indexes]

                loss, _ = session.run([loss_function, train_step], feed_dict={
                    premise_input: batch_premises,
                    hypothesis_input: batch_hypotheses,
                    label_input: batch_labels,
                    dropout_input: args.rnn_dropout_ratio
                })
                progress.update(batch_index, [("Loss", loss)])
                epoch_loss += loss
                batch_index += 1
            print("Current mean training loss: {}\n".format(epoch_loss / num_batches))

            print("-- Validating model")
            dev_num_examples = dev_labels.shape[0]
            dev_batches_indexes = np.arange(dev_num_examples)
            dev_num_correct = 0

            for start_idx in range(0, dev_num_examples - args.batch_size + 1, args.batch_size):
                dev_batch_indexes = dev_batches_indexes[start_idx:start_idx + args.batch_size]
                dev_batch_premises = dev_premises[dev_batch_indexes]
                dev_batch_hypotheses = dev_hypotheses[dev_batch_indexes]
                dev_batch_labels = dev_labels[dev_batch_indexes]
                predictions = session.run(
                    tf.argmax(logits, axis=1),
                    feed_dict={
                        premise_input: dev_batch_premises,
                        hypothesis_input: dev_batch_hypotheses,
                        dropout_input: 1.0
                    }
                )
                dev_num_correct += (predictions == dev_batch_labels).sum()
            dev_accuracy = dev_num_correct / dev_num_examples
            print("Current mean validation accuracy: {}".format(dev_accuracy))

            if dev_accuracy > dev_best_accuracy:
                stopping_step = 0
                best_epoch = epoch + 1
                dev_best_accuracy = dev_accuracy
                saver.save(session, args.model_save_filename + ".ckpt")
                print("Best mean validation accuracy: {} (reached at epoch {})".format(dev_best_accuracy, best_epoch))
                print("Best model saved to: {}".format(args.model_save_filename))
            else:
                stopping_step += 1
                print("Current stopping step: {}".format(stopping_step))
            if stopping_step >= args.patience:
                print("Early stopping at epoch {}!".format(epoch + 1))
                print("Best mean validation accuracy: {} (reached at epoch {})".format(dev_best_accuracy, best_epoch))
                should_stop = True
            if epoch + 1 >= args.num_epochs:
                print("Stopping at epoch {}!".format(epoch + 1))
                print("Best mean validation accuracy: {} (reached at epoch {})".format(dev_best_accuracy, best_epoch))
