# Experiments 2
This folder contains a batch of experiments performed to solve the Grounded Textual Entailment task.

## Stage I
At this stage, the following models are evaluated on the Grounded Textual Entailment task:

- **Simple TE (Blind)**: implementation of a model inspired by the baselines proposed in [1] to solve the Visual Question Answering task and in [2] to solve the Textual Entailment task. The model is a Siamese architecture which separately encodes both the premise and the hypothesis by using a recurrent neural network with long short-term memory units to obtain two vectors of 512 dimensions representing the two sentences. Then, the model feeds the concatenation of the two vectors to a stack of three layers of 512 dimensions having a Gated Hyperbolic Tangent (gated tanh) activation function and exploits a last layer having the softmax activation function to classify the relation between the two sentences as entailment, contradiction or neutral. The model exploits the 300000 most-frequent pre-trained GloVe embeddings and improves them during the training process. Dropout is applied to the inputs and outputs of the recurrent layers with a keep rate of 0.5 to regularize the model. Moreover, L2 regularization is applied to all the parameters of the model with a strength coefficient of 0.000005. The model is trained by using the Adam optimizer with a learning rate of 0.001 until the performance on the development set decreases for 3 times.
The training script is called train_simple_te_model.py, whereas the evaluation script is called eval_simple_te_model.py.
![image](https://raw.githubusercontent.com/hoavt-54/nli-images/master/models/images/Simple%20(Blind).png)
The detailed results of the model at this stage are reported at [https://github.com/hoavt-54/nli-images/tree/master/models/results/simple_te_model](https://github.com/hoavt-54/nli-images/tree/master/models/results/simple_te_model).
- **Simple VTE + CNN**: extension of the previous model which exploits an image associated to each pair of sentences. The model separately encodes both the premise and the hypothesis by using a recurrent neural network with long short-term memory units to obtain two vectors of 512 dimensions representing the two sentences. Then, a fully-connected layer receives the L2-normalized image vector of 2048 dimensions coming from the penultimate layer of a ResNet-101 convolutional neural network which receives the image as input to obtain a reduced vector of 512 dimensions. A fully-connected layer with a gated tanh activation function is also applied to the premise and to the hypothesis in order to obtain a reduced vector of 512 dimensions for each of them. The multimodal fusion between the premise (hypothesis) and the image is obtained by performing an element-wise multiplication between the reduced vector of the premise (hypothesis) and the reduced vector of the image. After that, the model feeds the concatenation of the two resulting multimodal representations to a stack of three layers of 512 dimensions having a gated tanh activation function and exploits a last layer having the softmax activation function to classify the relation between the two sentences as entailment, contradiction or neutral. The model exploits the 300000 most-frequent pre-trained GloVe embeddings and improves them during the training process. Dropout is applied to the inputs and outputs of the recurrent layers with a keep rate of 0.5 to regularize the model. Moreover, L2 regularization is applied to all the parameters of the model with a strength coefficient of 0.000005. The model is trained by using the Adam optimizer with a learning rate of 0.001 until the performance on the development set decreases for 3 times.
The training script is called train_simple_vte_model.py, whereas the evaluation script is called eval_simple_vte_model.py.
![image](https://raw.githubusercontent.com/hoavt-54/nli-images/master/models/images/Simple%20%2B%20CNN.png)
The detailed results of the model at this stage are reported at [https://github.com/hoavt-54/nli-images/tree/master/models/results/simple_vte_model](https://github.com/hoavt-54/nli-images/tree/master/models/results/simple_vte_model).
- **Bottom-up top-down attention VTE**: implementation of the model proposed in [3, 4], which was adapted to deal with the Grounded Textual Entailment task. The model separately encodes both the premise and the hypothesis by using a recurrent neural network with long short-term memory units to obtain two vectors of 512 dimensions representing the two sentences. A bottom-up attention mechanism using a Faster R-CNN based on a ResNet-101 convolutional neural network is performed to obtain region proposals of the 36 most informative regions of the image. A top-down attention mechanism between the premise (hypothesis) and the image is performed on each of the 36 L2-normalized image vectors of 2048 dimensions associated to each of the 36 region proposals to obtain an attention score for each of the regions. Then, a image vector of 2048 dimensions encoding the most interesting visual features for the premise (hypothesis) is obtained as a sum of the 36 image vectors weighted by the corresponding attention scores for the premise (hypothesis). A fully-connected layer with a gated tanh activation function is applied to the image vector of the most interesting visual features for the premise and for the hypothesis to obtain a reduced vector of 512 dimensions for each of them. A fully-connected layer with a gated tanh activation function is also applied to the premise and to the hypothesis in order to obtain a reduced vector of 512 dimensions for each of them. The multimodal fusion between the premise (hypothesis) and the image vector of the most interesting visual features for the premise (hypothesis) is obtained by performing an element-wise multiplication between the reduced vector of the premise (hypothesis) and the reduced vector of the most interesting visual features for the premise (hypothesis). After that, the model feeds the concatenation of the two resulting multimodal representations to a stack of three layers of 512 dimensions having a gated tanh activation function and exploits a last layer having the softmax activation function to classify the relation between the two sentences as entailment, contradiction or neutral. The model exploits the 300000 most-frequent pre-trained GloVe embeddings and improves them during the training process. Dropout is applied to the inputs and outputs of the recurrent layers with a keep rate of 0.5 to regularize the model. Moreover, L2 regularization is applied to all the parameters of the model with a strength coefficient of 0.000005. The model is trained by using the Adam optimizer with a learning rate of 0.001 until the performance on the development set decreases for 3 times.
The training script is called train_bottom_up_top_down_vte_model.py, whereas the evaluation script is called eval_bottom_up_top_down_vte_model.py.
![image](https://raw.githubusercontent.com/hoavt-54/nli-images/master/models/images/Bottom-up%20top-down.png)
The detailed results of the model at this stage are reported at [https://github.com/hoavt-54/nli-images/tree/master/models/results/bottom_up_top_down_vte_model](https://github.com/hoavt-54/nli-images/tree/master/models/results/bottom_up_top_down_vte_model).

The results of the stage I are reported in the following table:

| TRAINING SET  | TEST SET           | Simple TE (Blind) | Simple VTE + CNN | Bottom-up top-down VTE |
|---------------|--------------------|-------------------|------------------|------------------------|
| (V)SNLI train | (V)SNLI test       | 80.96             | 78.59            | 78.25                  |
| (V)SNLI train | (V)SICK2           | 56.05             | 55.66            | 52.31                  |
| (V)SNLI train | (V)SICK2 difficult | 45.16             | 50.36            | 47.5                   |

Moreover, the results of the stage I for each class are reported in the following table:

| TRAINING SET  | TEST SET           | Simple TE (Blind)                                                                  | Simple VTE + CNN                                                            | Bottom-up top-down                                                      |
|---------------|--------------------|------------------------------------------------------------------------------------|-----------------------------------------------------------------------------|-------------------------------------------------------------------------|
| (V)SNLI train | (V)SNLI test       | Neutral: 77.1, Entailment: 85.42, Contradiction: 80.17, Overall: 80.96             | Neutral: 74.87, Entailment: 83.94, Contradiction: 76.74, Overall: 78.59     | Neutral: 72.47, Entailment: 83.34, Contradiction: 78.68, Overall: 78.25 |
| (V)SNLI train | (V)SICK2           | Neutral: 39.35, Entailment: 85.5, Contradiction: 87.25, Overall: 56.05             | Neutral: 38.79, Entailment: 86.95, Contradiction: 82.52, Overall: 55.66     | Neutral: 35.61, Entailment: 80.35, Contradiction: 85.44, Overall: 52.31 |
| (V)SNLI train | (V)SICK2 difficult | Neutral: 19.35, Entailment: 67.31, Contradiction: 62.74, Overall: 45.16            | Neutral: 20.49, Entailment: 77.36, Contradiction: 65.38, Overall: 50.36     | Neutral: 22.95, Entailment: 65.09, Contradiction: 69.23 Overall: 47.5   |

## Stage II
At this stage, models are evaluated on the task of recognizing whether a sentence is a good caption for an image to identify the best multimodal fusion mechanism able to properly integrate language and vision. The considered models are the same evaluated at the stage II except for the fact that the last softmax layer evaluates a probability distribution among two classes, corresponding to the yes or no labels, instead of three. Moreover, L2 regularization has been applied, but dropout has been applied to all the fully-connected layers, as well as to the recurrent connection of the LSTMs. This choice has been motivated by the fact that L2 regularization did not allow the models to converge at this stage. The adapted models which corresponds to the models called **Simple TE**, **Simple VTE + CNN**, and **Bottom-up top-down VTE** have been called **Simple IC**, **Simple IC + CNN**, and **Bottom-up top-down IC**. At this stage, the models have been evaluated on a dataset  properly built to solve the described task and available at [https://github.com/hoavt-54/nli-images/tree/master/datasets/IC](https://github.com/hoavt-54/nli-images/tree/master/datasets/IC).

The results of the stage II are reported in the following table:

| Class | Simple IC (Blind) | Simple IC + CNN | Bottom-up top-down IC |
|-------|-------------------|-----------------|-----------------------|
| YES   | 83.45             | 96.56           | 98.09                 |
| NO    | 16.36             | 93.5            | 94.52                 |
| TOTAL | 49.91             | 95.03           | 96.31                 |

## Stage III
At this stage, a model exploiting the best-performing multimodal fusion mechanism identified in stage II is trained and evaluated in two different settings:

- Multi-task learning: the model is trained to solve the task of recognizing whether a sentence is a good caption for an image and the Grounded Textual Entailment task.
- Transfer learning: the model is first trained on the task of recognizing whether a sentence is a good caption for an image and then trained on the Grounded Textual Entailment task.

![image](https://raw.githubusercontent.com/hoavt-54/nli-images/master/models/images/Multi-Task%20or%20Transfer%20Learning.png)

The results of the stage III are reported in the following tables:

*Trained on IC and VSNLI train and evaluated on VSNLI test:*

| Class         | Simple TE (Blind) | Simple VTE | Top-down bottom-up attention | Multi-task learning | BiMPM | V-BiMPM |
|---------------|-------------------|------------|------------------------------|---------------------|-------|---------|
| ENTAILMENT    | 86.1              | 85.81      | 86.46                        | 85.54               | 90.03 | 90.38   |
| CONTRADICTION | 81.68             | 75.5       | 75.69                        | 75.44               | 86.25 | 87.53   |
| NEUTRAL       | 75.58             | 71.01      | 69.74                        | 71.36               | 82.79 | 82.91   |
| Total         | 81.2              | 77.56      | 77.43                        | 77.56               |       |         |

*Trained on IC and VSNLI train and evaluated on VSICK2:*

| Class         | Simple TE (Blind) | Simple VTE | Top-down bottom-up attention | Multi-task learning |
|---------------|-------------------|------------|------------------------------|---------------------|
| ENTAILMENT    | 88.79             | 85.04      | 83.58                        | 84.75               |
| CONTRADICTION | 85.29             | 84.79      | 82.2                         | 78.96               |
| NEUTRAL       | 35.06             | 27.13      | 26.85                        | 25.03               |
| Total         | 53.89             | 47.98      | 47.14                        | 45.91               |

*Trained on IC and VSNLI train and evaluated on Difficult VSICK2:*

| Class         | Simple TE (Blind) | Simple VTE | Top-down bottom-up attention | Multi-task learning |
|---------------|-------------------|------------|------------------------------|---------------------|
| ENTAILMENT    | 73.08             | 75.47      | 74.53                        | 78.3                |
| CONTRADICTION | 70.59             | 67.31      | 63.46                        | 65.38               |
| NEUTRAL       | 24.19             | 21.31      | 19.67                        | 18.03               |
| Total         | 50.9              | 50.36      | 48.57                        | 49.64               |

# References
[1] Bowman, Samuel R., et al. "A large annotated corpus for learning natural language inference." arXiv preprint arXiv:1508.05326 (2015).

[2] Antol, Stanislaw, et al. "Vqa: Visual question answering." Proceedings of the IEEE International Conference on Computer Vision. 2015.

[3] Anderson, Peter, et al. "Bottom-up and top-down attention for image captioning and VQA." arXiv preprint arXiv:1707.07998 (2017).

[4] Teney, Damien, et al. "Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge." arXiv preprint arXiv:1708.02711 (2017).
