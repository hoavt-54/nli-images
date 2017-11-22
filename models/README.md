# Models
This folder contains two models evaluated on the Recognizing Textual Entailment and Recognizing Visual Textual Entailment tasks. In particular, the models are described as follows:
* Bowman et al.: implementation of the simple 100d LSTM RNN baseline proposed in [1]. The model separately encodes both the premise and the hypothesis by using a recurrent neural network with long short-term memory units to obtain two vectors of 100 dimensions representing the two sentences. Then, the model feeds the concatenation of the two vectors to a stack of three layers of 200 dimensions with the tanh activation function and exploits a last layer with the softmax activation function to classify the relation between the two sentences as entailment, contradiction or neutral. The model exploits the pre-trained GloVe embeddings ([http://nlp.stanford.edu/data/glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip)) and improves them during the training process. Dropout is applied to the inputs and outputs of the recurrent layers with a rate of 0.2 to regularize the model. Moreover, L2 regularization is applied to all the parameters of the model with a strength coefficient of 0.0001. The model is trained by using the Adadelta optimizer until the performance on the development set stops improving. train_bowman_te.py is the training script, whereas evaluate_bowman_te.py is the evaluation script.
* Bowman et al. + images: extension of the previous model which exploits an image associated to each pair of premise and hypothesis to evaluate whether the performance improves with respect to using only the textual information. The only difference between this model and the previous one is that the image vector of size 4096 coming from the penultimate of a VGG-16 convolutional neural network which receives the image as input is concatenated to the vectors of the premise and the hypothesis before the classification process.

## Results of the Recognizing Textual Inference task

### Without transfer learning:

| train dataset | test dataset | Bowman et al. | BiMPM |
|---------------|--------------|---------------|-------|
| SNLI train    | SNLI test    | 78.67         | 86.39 |
| SNLI train    | SICK test    | 52.26         | 39.93 |
| SNLI train    | SICK2        | 56.86         | 56.49 |
| SICK train    | SICK test    | 56.87         | 83.33 |

### With transfer learning:

| pre-train dataset | train dataset | test dataset | Bowman et al. | BiMPM |
|-------------------|---------------|--------------|---------------|-------|
| SNLI train        | SICK train    | SICK test    | 77.46         |       |
| SNLI train        | SICK2 train   | SICK2 test   | 76.95         |       |

## Results of the Recognizing Visual Textual Inference task

### Without transfer learning:

| train dataset | test dataset | Bowman et al. + images | V-BiMPM |
|---------------|--------------|------------------------|---------|
| VSNLI train   | VSNLI test   | 33.82                  | 87.19   |
| VSNLI train   | VSICK2       |                        | 60.57   |

### With transfer learning:

| pre-train dataset | train dataset | test dataset | Bowman et al. | V-BiMPM |
|-------------------|---------------|--------------|---------------|---------|
| VSNLI train       | VSICK2 train  | VSICK2 test  |               |         |

[1] Bowman, Samuel R., et al. "A large annotated corpus for learning natural language inference.".
