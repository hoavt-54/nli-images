#!/bin/bash

# Recognizing Visual Textual Entailment with Transfer Learning

# Bowman et al. VSNLI train -> VSICK2 train -> VSICK2 test

~/python3 train_vte_baseline.py --train_filename=../datasets/snli_1.0/snli_1.0_train_filtered.tsv --dev_filename=../datasets/snli_1.0/snli_1.0_dev_filtered.tsv --vectors_filename=../../pre-wordvec/glove.840B.300d.txt --img_names_filename=../../flickr30k-cnn/flickr30k/filenames_77512.json --img_features_filename=../../flickr30k-cnn/flickr30k/vgg_feats_77512.npy --model_load_filename=checkpoints/vte/vsnli_train --model_save_filename=checkpoints/vte/vsnli_train_to_VSICK2_train

~/python3 eval_vte_baseline.py --model_filename=checkpoints/vte/vsnli_train --test_filename=../datasets/snli_1.0/snli_1.0_test_filtered.tsv --img_names_filename=../../flickr30k-cnn/flickr30k/filenames_77512.json --img_features_filename=../../flickr30k-cnn/flickr30k/vgg_feats_77512.npy --result_filename=results/vte/vsnli_train_to_VSICK2_train_to_VSICK2_test.txt
