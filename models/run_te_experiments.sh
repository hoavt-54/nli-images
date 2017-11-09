#!/bin/bash

# Recognizing Textual Entailment

# Bowman et al. SNLI train -> SNLI test

~/python3 train_te_baseline.py --train_filename=../datasets/snli_1.0/snli_1.0_train_filtered.tsv --dev_filename=../datasets/snli_1.0/snli_1.0_dev_filtered.tsv --vectors_filename=../../pre-wordvec/glove.840B.300d.txt --model_save_filename=checkpoints/te/snli_train
echo -en '\n'

~/python3 eval_te_baseline.py --model_filename=checkpoints/te/snli_train --test_filename=../datasets/snli_1.0/snli_1.0_test_filtered.tsv
echo -en '\n'

# Bowman et al. SNLI train -> SICK test

~/python3 eval_te_baseline.py  --model_filename=checkpoints/te/snli_train --test_filename=../datasets/SICK/SICK/SICK_test.tsv
echo -en '\n'

# Bowman et al. SNLI train -> SICK2

~/python3 eval_te_baseline.py --model_filename=checkpoints/te/snli_train --test_filename=../datasets/SICK/SICK2/SICK2.tsv
echo -en '\n'

# Bowman et al. SICK train -> SICK test

~/python3 train_te_baseline.py --train_filename=../datasets/SICK/SICK/SICK_train.tsv --dev_filename=../datasets/SICK/SICK/SICK_dev.tsv --vectors_filename=../../pre-wordvec/glove.840B.300d.txt --model_save_filename=checkpoints/te/SICK_train
echo -en '\n'

~/python3 eval_te_baseline.py --model_filename=checkpoints/te/SICK_train --test_filename=../datasets/SICK/SICK/SICK_test.tsv
echo -en '\n'

## Results of the Recognizing Textual Inference task
The results of the Recognizing Textual Inference task are reported in the following table:

| train dataset | test dataset | Bowman et al. | BiMPM |
|---------------|--------------|---------------|-------|
| SNLI train    | SNLI test    |               |       |
| SNLI train    | SICK test    |               |       |
| SNLI train    | SICK2        |               |       |
| SICK train    | SICK test    |               |       |

## Results of the Recognizing Visual Textual Inference task
The results of the Recognizing Visual Textual Inference task are reported in the following table:

| train dataset | test dataset | Bowman et al. + images | V-BiMPM |
|---------------|--------------|------------------------|---------|
| VSNLI train   | VSNLI test   |                        |         |
| VSNLI train   | VSICK        |                        |         |

