#!/bin/bash

# Recognizing Textual Entailment with Transfer Learning

~/python3 eval_bowman_te_baseline.py --model_filename=checkpoints/bowman_TE/SNLI_train --test_filename=../datasets/SICK/SICK2/difficult_SICK2.tsv --result_filename=results/bowman_TE/SNLI_train_to_difficult_SICK2

~/python3 eval_bowman_te_baseline.py --model_filename=checkpoints/bowman_TE/SICK_train --test_filename=../datasets/SICK/SICK2/difficult_SICK2.tsv --result_filename=results/bowman_TE/SICK_train_to_difficult_SICK2

~/python3 eval_lstm_te_baseline.py --model_filename=checkpoints/lstm_TE/SNLI_train --test_filename=../datasets/SICK/SICK2/difficult_SICK2.tsv --result_filename=results/lstm_TE/SNLI_train_to_difficult_SICK2

~/python3 eval_lstm_te_baseline.py --model_filename=checkpoints/lstm_TE/SICK_train --test_filename=../datasets/SICK/SICK2/difficult_SICK2.tsv --result_filename=results/lstm_TE/SICK_train_to_difficult_SICK2

# Recognizing Textual Entailment with Transfer Learning

~/python3 eval_bowman_te_baseline.py --model_filename=checkpoints/bowman_TE/SNLI_train_to_SICK_train --test_filename=../datasets/SICK/SICK2/difficult_SICK2.tsv --result_filename=results/bowman_TE/SNLI_train_to_SICK_train_to_difficult_SICK2

~/python3 eval_bowman_te_baseline.py --model_filename=checkpoints/bowman_TE/SNLI_train_to_SICK2_train --test_filename=../datasets/SICK/SICK2/difficult_SICK2.tsv --result_filename=results/bowman_TE/SNLI_train_to_SICK2_train_to_difficult_SICK2

~/python3 eval_lstm_te_baseline.py --model_filename=checkpoints/lstm_TE/SNLI_train_to_SICK_train --test_filename=../datasets/SICK/SICK2/difficult_SICK2.tsv --result_filename=results/lstm_TE/SNLI_train_to_SICK_train_to_difficult_SICK2

~/python3 eval_lstm_te_baseline.py --model_filename=checkpoints/lstm_TE/SNLI_train_to_SICK2_train --test_filename=../datasets/SICK/SICK2/difficult_SICK2.tsv --result_filename=results/lstm_TE/SNLI_train_to_SICK2_train_to_difficult_SICK2

# Recognizing Visual Textual Entailment without Transfer Learning

~/python3 eval_bowman_vte_baseline.py --model_filename=checkpoints/bowman_VTE/VSNLI_train --test_filename=../datasets/SICK/VSICK2/difficult_VSICK2.tsv --img_names_filename=../../flickr8k-cnn/flickr8k/filenames_4096.json --img_features_filename=../../flickr8k-cnn/flickr8k/vgg_feats_4096.npy --result_filename=results/bowman_VTE/VSNLI_train_to_difficult_VSICK2

~/python3 eval_lstm_vte_baseline.py --model_filename=checkpoints/lstm_VTE/VSNLI_train --test_filename=../datasets/SICK/VSICK2/difficult_VSICK2.tsv --img_names_filename=../../flickr8k-cnn/flickr8k/filenames_4096.json --img_features_filename=../../flickr8k-cnn/flickr8k/vgg_feats_4096.npy --result_filename=results/lstm_VTE/VSNLI_train_to_difficult_VSICK2

# Recognizing Visual Textual Entailment with Transfer Learning

~/python3 eval_bowman_vte_baseline.py --model_filename=checkpoints/bowman_VTE/VSNLI_train_to_VSICK2_train --test_filename=../datasets/SICK/VSICK2/difficult_VSICK2.tsv --img_names_filename=../../flickr8k-cnn/flickr8k/filenames_4096.json --img_features_filename=../../flickr8k-cnn/flickr8k/vgg_feats_4096.npy --result_filename=results/bowman_VTE/VSNLI_train_to_VSICK2_train_to_difficult_VSICK2

~/python3 eval_lstm_vte_baseline.py --model_filename=checkpoints/lstm_VTE/VSNLI_train_to_VSICK2_train --test_filename=../datasets/SICK/VSICK2/difficult_VSICK2.tsv --img_names_filename=../../flickr8k-cnn/flickr8k/filenames_4096.json --img_features_filename=../../flickr8k-cnn/flickr8k/vgg_feats_4096.npy --result_filename=results/lstm_VTE/VSNLI_train_to_VSICK2_train_to_difficult_VSICK2
