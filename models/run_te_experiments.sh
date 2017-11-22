#!/bin/bash

# Recognizing Textual Entailment  without Transfer Learning

# Bowman et al. SNLI train -> SNLI test

~/python3 train_te_baseline.py --train_filename=../datasets/snli_1.0/snli_1.0_train_filtered.tsv --dev_filename=../datasets/snli_1.0/snli_1.0_dev_filtered.tsv --vectors_filename=../../pre-wordvec/glove.840B.300d.txt --model_save_filename=checkpoints/te/snli_train

~/python3 eval_te_baseline.py --model_filename=checkpoints/te/snli_train --test_filename=../datasets/snli_1.0/snli_1.0_test_filtered.tsv --result_filename=results/te/snli_train_to_snli_test.txt

# Bowman et al. SNLI train -> SICK test

~/python3 eval_te_baseline.py  --model_filename=checkpoints/te/snli_train --test_filename=../datasets/SICK/SICK/SICK_test.tsv --result_filename=results/te/snli_train_to_SICK_test.tsv

# Bowman et al. SNLI train -> SICK2

~/python3 eval_te_baseline.py --model_filename=checkpoints/te/snli_train --test_filename=../datasets/SICK/SICK2/SICK2.tsv --result_filename=results/te/snli_train_to_SICK2.tsv

# Bowman et al. SICK train -> SICK test

~/python3 train_te_baseline.py --train_filename=../datasets/SICK/SICK/SICK_train.tsv --dev_filename=../datasets/SICK/SICK/SICK_dev.tsv --vectors_filename=../../pre-wordvec/glove.840B.300d.txt --model_save_filename=checkpoints/te/SICK_train

~/python3 eval_te_baseline.py --model_filename=checkpoints/te/SICK_train --test_filename=../datasets/SICK/SICK/SICK_test.tsv --result_filename=results/te/SICK_train_to_SICK_test.tsv
