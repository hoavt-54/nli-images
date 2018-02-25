~/python3 ../train_simple_te_model.py --train_filename=../../datasets/snli_1.0/snli_1.0_train_filtered.tokens --dev_filename=../../datasets/snli_1.0/snli_1.0_dev_filtered.tokens --vectors_filename=../../../pre-wordvec/glove.840B.300d.txt --model_save_filename=../checkpoints/simple_te_model/snli_train
~/python3 ../eval_simple_te_model.py --model_filename=../checkpoints/simple_te_model/snli_train --test_filename=../../datasets/snli_1.0/snli_1.0_test_filtered.tokens --result_filename=../results/simple_te_model/snli_train_to_snli_test
~/python3 ../eval_simple_te_model.py  --model_filename=../checkpoints/simple_te_model/snli_train --test_filename=../datasets/SICK/SICK2/SICK2.tokens --result_filename=../results/simple_te_model/snli_train_to_sick2
~/python3 ../eval_simple_te_model.py  --model_filename=../checkpoints/simple_te_model/snli_train --test_filename=../datasets/SICK/SICK2/difficult_SICK2.tokens --result_filename=../results/simple_te_model/snli_train_to_difficult_sick2
