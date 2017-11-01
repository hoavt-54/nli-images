import errno
import glob
import json
import os
import random
import time
from argparse import ArgumentParser

import dill as pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as O
from torch.autograd import Variable
from torchtext import data


def makedirs(name):
    try:
        os.makedirs(name)
    except OSError as ex:
        if ex.errno == errno.EEXIST and os.path.isdir(name):
            pass
        else:
            raise


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
        self._rnn = nn.LSTM(input_size=embedding_size, hidden_size=rnn_hidden_size, dropout=rnn_dropout_ratio)

    def forward(self, inputs):
        batch_size = inputs.size()[1]
        state_shape = 1, batch_size, self._rnn_hidden_size
        h0 = c0 = Variable(inputs.data.new(*state_shape).zero_())
        outputs, (ht, ct) = self._rnn(inputs, (h0, c0))
        return ht[-1]


class VSNLIClassifier(nn.Module):
    def __init__(self, num_tokens, num_labels, embedding_size, fix_embeddings, rnn_hidden_size, rnn_dropout_ratio, img_names_features, images_vocab):
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


if __name__ == "__main__":
    random.seed(1235)
    torch.manual_seed(1235)
    parser = ArgumentParser(description="NLI")
    parser.add_argument("--train_filename", type=str, required=True)
    parser.add_argument("--dev_filename", type=str, required=True)
    parser.add_argument("--test_filename", type=str, required=True)
    parser.add_argument("--img_names_filename", type=str, required=True)
    parser.add_argument("--img_features_filename", type=str, required=True)
    parser.add_argument("--index_filename", type=str, required=True)
    parser.add_argument("--embedding_size", type=int, default=300)
    parser.add_argument("--fix_embeddings", type=bool, default=False)
    parser.add_argument("--rnn_hidden_size", type=int, default=100)
    parser.add_argument("--rnn_dropout_ratio", type=float, default=0.2)
    parser.add_argument("--word_vectors", type=str, default="glove.840B.300d")
    parser.add_argument("--vector_cache", type=str, default=os.path.join(os.getcwd(), ".vector_cache/input_vectors.pt"))
    parser.add_argument("--data_cache", type=str, default=os.path.join(os.getcwd(), ".data_cache"))
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1.0)
    parser.add_argument("--l2_reg", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--resume_snapshot", type=str, default="")
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--dev_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--save_path", type=str, default="results")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    inputs = data.Field(lower=True)
    answers = data.Field(sequential=False, unk_token=None, preprocessing=lambda x: x.strip())
    images = data.Field(sequential=False, unk_token=None, preprocessing=lambda x: x.strip())

    train = data.TabularDataset(
        path=args.train_filename,
        format="tsv",
        fields=[
            ("label", answers),
            ("premise", inputs),
            ("hypothesis", inputs),
            ("image", images)
        ]
    )
    dev = data.TabularDataset(
        path=args.dev_filename,
        format="tsv",
        fields=[
            ("label", answers),
            ("premise", inputs),
            ("hypothesis", inputs),
            ("image", images)
        ]
    )
    test = data.TabularDataset(
        path=args.test_filename,
        format="tsv",
        fields=[
            ("label", answers),
            ("premise", inputs),
            ("hypothesis", inputs),
            ("image", images)
        ]
    )

    inputs.build_vocab(train, dev, test)
    if args.word_vectors:
        if os.path.isfile(args.vector_cache):
            inputs.vocab.vectors = torch.load(args.vector_cache)
        else:
            inputs.vocab.load_vectors(args.word_vectors)
            makedirs(os.path.dirname(args.vector_cache))
            torch.save(inputs.vocab.vectors, args.vector_cache)
    answers.build_vocab(train, dev, test)
    images.build_vocab(train, dev, test)

    with open(args.index_filename, mode="wb") as out_file:
        pickle.dump({"inputs": inputs, "answers": answers}, out_file)

    with open(args.img_names_filename) as in_file:
        img_names = json.load(in_file)

    with open(args.img_features_filename) as in_file:
        img_features = np.load(in_file).reshape(-1, 49, 512)

    img_names_features = {filename: features for filename, features in zip(img_names, img_features)}

    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test),
        batch_size=args.batch_size,
        repeat=False,
        device=args.gpu
    )

    num_tokens = len(inputs.vocab)
    num_labels = len(answers.vocab)

    if args.resume_snapshot:
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))
    else:
        model = VSNLIClassifier(
            num_tokens,
            num_labels,
            args.embedding_size,
            args.fix_embeddings,
            args.rnn_hidden_size,
            args.rnn_dropout_ratio,
            img_names_features,
            images.vocab
        )
        if args.word_vectors:
            model._embedding.weight.data = inputs.vocab.vectors
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    opt = O.Adadelta(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)

    iterations = 0
    start = time.time()
    best_dev_acc = -1
    stopping_step = 0
    header = "  Time Epoch Iteration Progress    (%Epoch)   Loss  Train/Accuracy  Dev/Accuracy"
    dev_log_template = ' '.join("{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:8.6f},{:12.4f},{:12.4f}".split(","))
    log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
    makedirs(args.save_path)
    print(header)

    for epoch in range(args.num_epochs):
        train_iter.init_epoch()
        n_correct, n_total = 0, 0

        for batch_idx, batch in enumerate(train_iter):
            iterations += 1
            model.train()
            opt.zero_grad()
            answer = model(batch)

            n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
            n_total += batch.batch_size
            train_acc = 100. * n_correct / n_total

            loss = criterion(answer, batch.label)
            loss.backward()
            opt.step()

            if iterations % args.log_every == 0:
                print(
                    log_template.format(
                        time.time() - start,
                        epoch, iterations,
                        1 + batch_idx,
                        len(train_iter),
                        100. * (1 + batch_idx) / len(train_iter),
                        loss.data[0],
                        " " * 8,
                        n_correct / n_total * 100,
                        " " * 12
                    )
                )

        if epoch % args.save_every == 0:
            snapshot_prefix = os.path.join(args.save_path, "snapshot")
            snapshot_path = snapshot_prefix + "_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt".format(
                train_acc,
                loss.data[0],
                iterations
            )
            torch.save(model, snapshot_path)
            for f in glob.glob(snapshot_prefix + '*'):
                if f != snapshot_path:
                    os.remove(f)

        if epoch % args.dev_every == 0:
            model.eval()
            dev_iter.init_epoch()

            n_dev_correct, dev_loss = 0, 0
            for dev_batch_idx, dev_batch in enumerate(dev_iter):
                answer = model(dev_batch)
                n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                dev_loss = criterion(answer, dev_batch.label)
            dev_acc = 100. * n_dev_correct / len(dev)

            print(
                dev_log_template.format(
                    time.time() - start,
                    epoch,
                    iterations,
                    1 + batch_idx,
                    len(train_iter),
                    100. * (1 + batch_idx) / len(train_iter),
                    loss.data[0],
                    train_acc,
                    dev_acc
                )
            )

            if dev_acc > best_dev_acc:
                stopping_step = 0
                best_dev_acc = dev_acc
                snapshot_prefix = os.path.join(args.save_path, "best_snapshot")
                snapshot_path = snapshot_prefix + "_devacc_{}_devloss_{}__epoch_{}_model.pt".format(
                    dev_acc,
                    dev_loss.data[0],
                    epoch
                )

                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

            else:
                stopping_step += 1

            if stopping_step >= args.patience:
                exit(0)
