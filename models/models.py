import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable


class Bottle(nn.Module):
    def forward(self, input):
        if len(input.size()) <= 2:
            return super(Bottle, self).forward(input)
        size = input.size()[:2]
        out = super(Bottle, self).forward(input.view(size[0] * size[1], -1))
        return out.view(size[0], size[1], -1)


class Linear(Bottle, nn.Linear):
    pass


class Encoder(nn.Module):
    def __init__(self, embedding_size, rnn_hidden_size, rnn_dropout_ratio):
        super(Encoder, self).__init__()
        self._embedding_size = embedding_size
        self._rnn_hidden_size = rnn_hidden_size
        self._rnn_dropout_ratio = rnn_dropout_ratio
        self._rnn = nn.LSTM(embedding_size, rnn_hidden_size, dropout=rnn_dropout_ratio)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = 1, batch_size, self._rnn_hidden_size
        h0 = c0 = Variable(inputs.data.new(*state_shape).zero_())
        outputs, (ht, ct) = self._rnn(inputs, (h0, c0))
        return ht[-1]


class SNLIClassifier(nn.Module):
    def __init__(self,
                 num_tokens,
                 num_labels,
                 embedding_size,
                 fix_embeddings,
                 rnn_hidden_size,
                 rnn_dropout_ratio):
        super(SNLIClassifier, self).__init__()
        self._num_tokens = num_tokens
        self._num_labels = num_labels
        self._embedding_size = embedding_size
        self._fix_embeddings = fix_embeddings
        self._rnn_hidden_size = rnn_hidden_size
        self._rnn_dropout_ratio = rnn_dropout_ratio
        self._embedding = nn.Embedding(num_tokens, embedding_size)
        self._encoder = Encoder(embedding_size, rnn_hidden_size, rnn_dropout_ratio)
        self._dropout = nn.Dropout(rnn_dropout_ratio)
        self._tanh = nn.Tanh()
        seq_in_size = 2 * rnn_hidden_size
        lin_config = [seq_in_size] * 2
        self.out = nn.Sequential(
            Linear(*lin_config),
            self._tanh,
            self._dropout,
            Linear(*lin_config),
            self._tanh,
            self._dropout,
            Linear(*lin_config),
            self._tanh,
            self._dropout,
            Linear(seq_in_size, num_labels)
        )

    def forward(self, batch):
        premise_embedding = self._embedding(batch.premise)
        hypothesis_embedding = self._embedding(batch.hypothesis)
        if self._fix_embeddings:
            premise_embedding = Variable(premise_embedding.data)
            hypothesis_embedding = Variable(hypothesis_embedding.data)
        premise = self._encoder(premise_embedding)
        hypothesis = self._encoder(hypothesis_embedding)
        scores = self.out(torch.cat([premise, hypothesis], 1))
        return scores


class VSNLIClassifier(nn.Module):
    def __init__(self, num_tokens, num_labels, embedding_size, fix_embeddings, rnn_hidden_size, rnn_dropout_ratio,
                 img_names_features, images_vocab):
        super(VSNLIClassifier, self).__init__()
        self._num_tokens = num_tokens
        self._num_labels = num_labels
        self._embedding_size = embedding_size
        self._fix_embeddings = fix_embeddings
        self._rnn_hidden_size = rnn_hidden_size
        self._rnn_dropout_ratio = rnn_dropout_ratio
        self._img_names_features = img_names_features
        self._images_vocab = images_vocab
        self._embedding = nn.Embedding(num_tokens, embedding_size)
        self._encoder = Encoder(embedding_size, rnn_hidden_size, rnn_dropout_ratio)
        self._dropout = nn.Dropout(p=rnn_dropout_ratio)
        self._tanh = nn.Tanh()
        seq_in_size = 2 * rnn_hidden_size + 512
        lin_config = [seq_in_size] * 2
        self.out = nn.Sequential(
            Linear(*lin_config),
            self._tanh,
            self._dropout,
            Linear(*lin_config),
            self._tanh,
            self._dropout,
            Linear(*lin_config),
            self._tanh,
            self._dropout,
            Linear(seq_in_size, num_labels)
        )

    def forward(self, batch):
        premise_embedding = self._embedding(batch.premise)
        hypothesis_embedding = self._embedding(batch.hypothesis)
        if self._fix_embeddings:
            premise_embedding = Variable(premise_embedding.data)
            hypothesis_embedding = Variable(hypothesis_embedding.data)
        premise = self._encoder(premise_embedding)
        hypothesis = self._encoder(hypothesis_embedding)
        features = [self._img_names_features[self._images_vocab.itos[name.data[0]]] for name in batch.image]
        features = Variable(torch.from_numpy(np.array(features)).cuda())
        features = torch.squeeze(torch.mean(features, 1))
        scores = self.out(torch.cat([premise, hypothesis, features], 1))
        return scores
