import atexit
import os
import random
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

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
