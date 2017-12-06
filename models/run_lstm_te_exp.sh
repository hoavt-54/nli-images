#!/bin/bash

# Recognizing Textual Entailment  without Transfer Learning

# LSTM SNLI train -> SNLI test

~/python3 train_lstm_te_baseline.py --train_filename=../datasets/SNLI_1.0/SNLI_1.0_train_filtered.tsv --dev_filename=../datasets/SNLI_1.0/SNLI_1.0_dev_filtered.tsv --vectors_filename=../../pre-wordvec/glove.840B.300d.txt --model_save_filename=checkpoints/lstm_TE/SNLI_train

~/python3 eval_lstm_te_baseline.py --model_filename=checkpoints/lstm_TE/SNLI_train --test_filename=../datasets/SNLI_1.0/SNLI_1.0_test_filtered.tsv --result_filename=results/lstm_TE/SNLI_train_to_SNLI_test

# LSTM SNLI train -> SICK test

~/python3 eval_lstm_te_baseline.py  --model_filename=checkpoints/lstm_TE/SNLI_train --test_filename=../datasets/SICK/SICK/SICK_test.tsv --result_filename=results/lstm_TE/SNLI_train_to_SICK_test

# LSTM SNLI train -> SICK2

~/python3 eval_lstm_te_baseline.py --model_filename=checkpoints/lstm_TE/SNLI_train --test_filename=../datasets/SICK/SICK2/SICK2.tsv --result_filename=results/lstm_TE/SNLI_train_to_SICK2

# LSTM SICK train -> SICK test

~/python3 train_lstm_te_baseline.py --train_filename=../datasets/SICK/SICK/SICK_train.tsv --dev_filename=../datasets/SICK/SICK/SICK_dev.tsv --vectors_filename=../../pre-wordvec/glove.840B.300d.txt --model_save_filename=checkpoints/lstm_TE/SICK_train

~/python3 eval_lstm_te_baseline.py --model_filename=checkpoints/lstm_TE/SICK_train --test_filename=../datasets/SICK/SICK/SICK_test.tsv --result_filename=results/lstm_TE/SICK_train_to_SICK_test
