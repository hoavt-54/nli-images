# Models
This folder contains two models evaluated on the Recognizing Textual Entailment and Recognizing Visual Textual Entailment tasks. In particular, the models are described as follows:

* Bowman et al.: implementation of the simple 100d LSTM RNN baseline proposed in [1]. The model is basically a Siamese architecture which separately encodes both the premise and the hypothesis by using a recurrent neural network with long short-term memory units to obtain two vectors of 100 dimensions representing the two sentences. Then, the model feeds the concatenation of the two vectors to a stack of three layers of 200 dimensions having the Hyperbolic Tangent (tanh) activation function and exploits a last layer having the softmax activation function to classify the relation between the two sentences as entailment, contradiction or neutral. The model exploits the 300000 most-frequent [pre-trained GloVe embeddings](http://nlp.stanford.edu/data/glove.840B.300d.zip) and improves them during the training process. Dropout is applied to the inputs and outputs of the recurrent layers with a rate of 0.2 to regularize the model. Moreover, L2 regularization is applied to all the parameters of the model with a strength coefficient of 0.0001. The model is trained by using the Adadelta optimizer which is executed until the performance on the development set decreases for 3 times. The training script is called train_bowman_te_baseline.py, whereas the evaluation script is called eval_bowman_te_baseline.py.
* Bowman et al. + images: extension of the previous model which exploits an image associated to each pair of sentences. The only difference between this model and the previous one is that a fully-connected layer receives the image vector of 4096 dimensions coming from the penultimate of a VGG-16 convolutional neural network which receives the image as input and to obtain a vector of 200 dimensions which is concatenated to the vectors of the premise and the hypothesis before the classification process. The weights of the VGG-16 convolutional neural network are frozen. The training script is called train_bowman_vte_baseline.py, whereas the evaluation script is called eval_bowman_vte_baseline.py.
* LSTM: implementation of the Bowman et al. model having Rectified Linear Units (ReLUs) as activation functions in the fully connected layers and using the Adam optimizer with a learning rate of 0.001.
* LSTM + images: implementation of the Bowman et al. + images model having ReLUs as activation functions in the fully connected layers and using the Adam optimizer with a learning rate of 0.001.

## Results of the Recognizing Textual Entailment task

### Without transfer learning:

| training dataset | test dataset    | Bowman | LSTM  | BiMPM |
|------------------|-----------------|--------|-------|-------|
| SNLI train       | SNLI test       | 78.81  | 79.81 | 86.41 |
| SNLI train       | SICK test       | 48.51  | 48.53 | 39.93 |
| SNLI train       | SICK2           | 51.76  | 54.55 | 56.49 |
| SNLI train       | SICK2 difficult | 45.88  | 48.39 |       |
| SICK train       | SICK test       | 56.87  | 62.45 | 83.33 |
| SICK train       | SICK2 difficult | 44.44  | 50.18 |       |

### With transfer learning:

| pre-training dataset | training dataset | test dataset    | Bowman | LSTM  | BiMPM |
|----------------------|------------------|-----------------|--------|-------|-------|
| SNLI train           | SICK train       | SICK test       | 76.23  | 79.96 |       |
| SNLI train           | SICK train       | SICK2 difficult | 56.99  | 58.42 |       |
| SNLI train           | SICK2 train      | SICK2 test      | 73.73  | 77.59 |       |
| SNLI train           | SICK3 train      | SICK2 difficult | 54.12  | 56.27 |       |

## Results of the Recognizing Visual Textual Entailment task

### Without transfer learning:

| training dataset | test dataset    | Bowman + images | LSTM + images | V-BiMPM |
|------------------|-----------------|-----------------|---------------|---------|
| VSNLI train      | VSNLI test      | 76.45           | 79.15         | 86.99   |
| VSNLI train      | VSICK2          | 54.06           | 57.26         | 60.57   |
| VSNLI train      | SICK2 difficult | 48.57           | 48.21         |         |

### With transfer learning:

| pre-training dataset | training dataset | test dataset    | Bowman + images | LSTM + images | V-BiMPM |
|----------------------|------------------|-----------------|-----------------|---------------|---------|
| VSNLI train          | VSICK2 train     | VSICK2 test     | 72.03           | 78.67         |         |
| VSNLI train          | VSICK2 train     | SICK2 difficult | 57.86           | 54.64         |         |

## Comparison of the comparable configurations:

### Without transfer learning:

| training set  | test set     | Bowman | Bowman + images | LSTM  | LSTM + images | BiMPM | V-BiMPM |
|---------------|--------------|--------|-----------------|-------|---------------|-------|---------|
| (V)SNLI train | (V)SNLI test | 78.81  | 76.45           | 79.81 | 79.15         | 86.41 | 86.99   |
| (V)SNLI train | (V)SICK2     | 51.76  | 54.06           | 54.55 | 57.26         | 56.49 | 60.57   |

### With transfer learning:

| pre-training set | training set   | test set      | Bowman | Bowman + images | LSTM  | LSTM + images | BiMPM | V-BiMPM |
|------------------|----------------|---------------|--------|-----------------|-------|---------------|-------|---------|
| (V)SNLI train    | (V)SICK2 train | (V)SICK2 test | 73.73  | 72.03           | 77.59 | 78.67         |       |         |

[1] Bowman, Samuel R., et al. "A large annotated corpus for learning natural language inference.".
