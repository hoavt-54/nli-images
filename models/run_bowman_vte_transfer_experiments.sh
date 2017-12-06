#!/bin/bash

# Recognizing Visual Textual Entailment with Transfer Learning

# Bowman et al. VSNLI train -> VSICK2 train -> VSICK2 test

~/python3 train_bowman_vte_baseline.py --train_filename=../datasets/SICK/VSICK2/VSICK2_train.tsv --dev_filename=../datasets/SICK/VSICK2/VSICK2_dev.tsv --vectors_filename=../../pre-wordvec/glove.840B.300d.txt --img_names_filename=../../flickr8k-cnn/flickr8k/filenames_4096.json --img_features_filename=../../flickr8k-cnn/flickr8k/vgg_feats_4096.npy --model_load_filename=checkpoints/VTE/VSNLI_train --model_save_filename=checkpoints/VTE/VSNLI_train_to_VSICK2_train

~/python3 eval_bowman_vte_baseline.py --model_filename=checkpoints/VTE/VSNLI_train_to_VSICK2_train --test_filename=../datasets/SICK/VSICK2/VSICK2_test.tsv --img_names_filename=../../flickr8k-cnn/flickr8k/filenames_4096.json --img_features_filename=../../flickr8k-cnn/flickr8k/vgg_feats_4096.npy --result_filename=results/VTE/VSNLI_train_to_VSICK2_train_to_VSICK2_test
