#!/bin/bash

# Recognizing Visual Textual Entailment without Transfer Learning

# Bottom up top down VSNLI train -> VSNLI test

~/python3 train_top_down_baseline.py --train_filename=../datasets/snli_1.0/snli_1.0_train_filtered.tsv --dev_filename=../datasets/snli_1.0/snli_1.0_dev_filtered.tsv --vectors_filename=../../pre-wordvec/glove.840B.300d.txt --img_names_filename=../../../flickr30k_bottom_up_img_names.json --img_features_filename=../../../flickr30k_bottom_up_img_feats.npy --model_save_filename=checkpoints/bottom_up_top_down/VSNLI_train

~/python3 eval_top_down_baseline.py --model_filename=checkpoints/bottom_up_top_down/VSNLI_train --test_filename=../datasets/snli_1.0/snli_1.0_test_filtered.tsv --img_names_filename=../../../flickr30k_bottom_up_img_names.json --img_features_filename=../../../flickr30k_bottom_up_img_feats.npy --result_filename=results/bottom_up_top_down/VSNLI_train_to_VSNLI_test

# Bottom up top down VSNLI train -> VSICK2

~/python3 eval_top_down_baseline.py --model_filename=checkpoints/bottom_up_top_down/VSNLI_train --test_filename=../datasets/SICK/VSICK2/VSICK2.tsv --img_names_filename=../../../flickr8k_bottom_up_img_names.json --img_features_filename=../../../flickr8k_bottom_up_img_feats.npy --result_filename=results/bottom_up_top_down/VSNLI_train_to_VSNLI_test
