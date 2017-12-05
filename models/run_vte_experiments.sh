#!/bin/bash

# Recognizing Visual Textual Entailment without Transfer Learning

# Bowman et al. VSNLI train -> VSNLI test

~/python3 train_vte_baseline.py --train_filename=../datasets/snli_1.0/snli_1.0_train_filtered.tsv --dev_filename=../datasets/snli_1.0/snli_1.0_dev_filtered.tsv --vectors_filename=../../pre-wordvec/glove.840B.300d.txt --img_names_filename=../../flickr30k-cnn/flickr30k/filenames_4096.json --img_features_filename=../../flickr30k-cnn/flickr30k/vgg_feats_4096.npy --model_save_filename=checkpoints/VTE/VSNLI_train

~/python3 eval_vte_baseline.py --model_filename=checkpoints/VTE/VSNLI_train --test_filename=../datasets/snli_1.0/snli_1.0_test_filtered.tsv --img_names_filename=../../flickr30k-cnn/flickr30k/filenames_4096.json --img_features_filename=../../flickr30k-cnn/flickr30k/vgg_feats_4096.npy --result_filename=results/VTE/VSNLI_train_to_VSNLI_test

# Bowman et al. VSNLI train -> VSICK2

~/python3 eval_vte_baseline.py --model_filename=checkpoints/VTE/VSNLI_train --test_filename=../datasets/SICK/VSICK2/VSICK2.tsv --img_names_filename=../../flickr8k-cnn/flickr8k/filenames_4096.json --img_features_filename=../../flickr8k-cnn/flickr8k/vgg_feats_4096.npy --result_filename=results/VTE/VSNLI_train_to_VSICK2
