import atexit
import json
import os
import pickle
import random
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf
from tl_mt_model import build_tl_mt_model_hphi

from datasets import load_ic_dataset, load_vte_dataset, ImageReader
from embeddings import load_glove
from utils import start_logger, stop_logger, Progbar, batch

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
    parser.add_argument("--dropout_ratio", type=float, default=0.5)
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

    ic_logits, vte_logits = build_tl_mt_model_hphi(
        sentence_input,
        premise_input,
        hypothesis_input,
        img_features_input,
        dropout_input,
        num_tokens,
        ic_num_labels,
        vte_num_labels,
        embeddings,
        args.embeddings_size,
        args.num_img_features,
        args.img_features_size,
        args.train_embeddings,
        args.rnn_hidden_size,
        args.classification_hidden_size,
        args.multimodal_fusion_hidden_size
    )
    ic_loss_function = tf.losses.sparse_softmax_cross_entropy(ic_label_input, ic_logits)
    ic_train_step = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(ic_loss_function)
    vte_loss_function = tf.losses.sparse_softmax_cross_entropy(vte_label_input, vte_logits)
    vte_train_step = tf.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(vte_loss_function)
    saver = tf.train.Saver()

    ic_num_examples = ic_train_labels.shape[0]
    vte_num_examples = vte_train_labels.shape[0]
    ic_num_batches = ic_num_examples // args.batch_size
    vte_num_batches = vte_num_examples // args.batch_size
    num_batches = ic_num_batches + vte_num_batches
    dev_best_accuracy = -1
    stopping_step = 0
    best_epoch = None
    should_stop = False

    with tf.Session(config=tf.ConfigProto(inter_op_parallelism_threads=1)) as session:
        session.run(tf.global_variables_initializer())

        for epoch in range(args.num_epochs):
            if should_stop:
                break

            print("\n==> Online epoch # {0}".format(epoch + 1))
            progress = Progbar(vte_num_batches)
            ic_batches_indexes = np.arange(ic_num_examples)
            np.random.shuffle(ic_batches_indexes)
            vte_batches_indexes = np.arange(vte_num_examples)
            np.random.shuffle(vte_batches_indexes)
            batch_index = 1
            vte_epoch_loss = 0

            ic_batches = batch(ic_batches_indexes, args.batch_size)
            vte_batches = batch(vte_batches_indexes, args.batch_size)

            next_ic_batches = next(ic_batches, None)
            next_vte_batches = next(vte_batches, None)

            while next_ic_batches is not None or next_vte_batches is not None:
                if next_ic_batches is not None:
                    ic_batch_sentences = ic_train_sentences[next_ic_batches]
                    ic_batch_labels = ic_train_labels[next_ic_batches]
                    ic_batch_img_names = [ic_train_img_names[i] for i in next_ic_batches]
                    ic_batch_img_features = ic_image_reader.get_features(ic_batch_img_names)

                    ic_loss, _ = session.run([ic_loss_function, ic_train_step], feed_dict={
                        sentence_input: ic_batch_sentences,
                        img_features_input: ic_batch_img_features,
                        ic_label_input: ic_batch_labels,
                        dropout_input: args.dropout_ratio
                    })
                    next_ic_batches = next(ic_batches, None)

                if next_vte_batches is not None:
                    vte_batch_premises = vte_train_premises[next_vte_batches]
                    vte_batch_hypotheses = vte_train_hypotheses[next_vte_batches]
                    vte_batch_labels = vte_train_labels[next_vte_batches]
                    vte_batch_img_names = [vte_train_img_names[i] for i in next_vte_batches]
                    vte_batch_img_features = vte_image_reader.get_features(vte_batch_img_names)

                    vte_loss, _ = session.run([vte_loss_function, vte_train_step], feed_dict={
                        premise_input: vte_batch_premises,
                        hypothesis_input: vte_batch_hypotheses,
                        img_features_input: vte_batch_img_features,
                        vte_label_input: vte_batch_labels,
                        dropout_input: args.dropout_ratio
                    })
                    vte_epoch_loss += vte_loss
                    progress.update(batch_index, [("Loss", vte_loss)])
                    next_vte_batches = next(vte_batches, None)
                    batch_index += 1

            print("Current mean training loss: {}\n".format(vte_epoch_loss / vte_num_batches))

            print("-- Validating model")
            dev_num_examples = vte_dev_labels.shape[0]
            dev_batches_indexes = np.arange(dev_num_examples)
            dev_num_correct = 0

            for vte_indexes in batch(dev_batches_indexes, args.batch_size):
                vte_dev_batch_premises = vte_dev_premises[vte_indexes]
                vte_dev_batch_hypotheses = vte_dev_hypotheses[vte_indexes]
                vte_dev_batch_labels = vte_dev_labels[vte_indexes]
                vte_dev_batch_img_names = [vte_dev_img_names[i] for i in vte_indexes]
                vte_dev_batch_img_features = vte_image_reader.get_features(vte_dev_batch_img_names)
                predictions = session.run(
                    tf.argmax(vte_logits, axis=1),
                    feed_dict={
                        premise_input: vte_dev_batch_premises,
                        hypothesis_input: vte_dev_batch_hypotheses,
                        img_features_input: vte_dev_batch_img_features,
                        dropout_input: 1.0
                    }
                )
                dev_num_correct += (predictions == vte_dev_batch_labels).sum()
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
