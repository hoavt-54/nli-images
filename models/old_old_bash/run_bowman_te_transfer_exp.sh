#!/bin/bash

# Recognizing Textual Entailment with Transfer Learning

# Bowman et al. SNLI train -> SICK train -> SICK test

~/python3 train_bowman_te_baseline.py --train_filename=../datasets/SICK/SICK/SICK_train.tsv --dev_filename=../datasets/SICK/SICK/SICK_dev.tsv --vectors_filename=../../pre-wordvec/glove.840B.300d.txt --model_load_filename=checkpoints/bowman_TE/SNLI_train --model_save_filename=checkpoints/bowman_TE/SNLI_train_to_SICK_train

~/python3 eval_bowman_te_baseline.py --model_filename=checkpoints/bowman_TE/SNLI_train_to_SICK_train --test_filename=../datasets/SICK/SICK/SICK_test.tsv --result_filename=results/bowman_TE/SNLI_train_to_SICK_train_to_SICK_test

# Bowman et al. SNLI train -> SICK2 train -> SICK2 test

~/python3 train_bowman_te_baseline.py --train_filename=../datasets/SICK/SICK2/SICK2_train.tsv --dev_filename=../datasets/SICK/SICK2/SICK2_dev.tsv --vectors_filename=../../pre-wordvec/glove.840B.300d.txt --model_load_filename=checkpoints/bowman_TE/SNLI_train --model_save_filename=checkpoints/bowman_TE/SNLI_train_to_SICK2_train

~/python3 eval_bowman_te_baseline.py --model_filename=checkpoints/bowman_TE/SNLI_train_to_SICK2_train --test_filename=../datasets/SICK/SICK2/SICK2_test.tsv --result_filename=results/bowman_TE/SNLI_train_to_SICK2_train_to_SICK2_test
