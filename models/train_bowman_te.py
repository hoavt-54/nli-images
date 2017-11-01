import glob
import os
import random
import time
from argparse import ArgumentParser

import dill as pickle
import torch
import torch.nn as nn
import torch.optim as O
from torchtext import data

from models.models import SNLIClassifier
from models.utils import makedirs

if __name__ == "__main__":
    random.seed(1235)
    torch.manual_seed(1235)
    parser = ArgumentParser()
    parser.add_argument("--train_filename", type=str, required=True)
    parser.add_argument("--dev_filename", type=str, required=True)
    parser.add_argument("--test_filename", type=str, required=True)
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

    train = data.TabularDataset(
        path=args.train_filename,
        format="tsv",
        fields=[
            ("label", answers),
            ("premise", inputs),
            ("hypothesis", inputs),
            ("images", None)
        ]
    )
    dev = data.TabularDataset(
        path=args.dev_filename,
        format="tsv",
        fields=[
            ("label", answers),
            ("premise", inputs),
            ("hypothesis", inputs),
            ("images", None)
        ]
    )
    test = data.TabularDataset(
        path=args.test_filename,
        format="tsv",
        fields=[
            ("label", answers),
            ("premise", inputs),
            ("hypothesis", inputs),
            ("images", None)
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

    with open(args.index_filename, mode="wb") as out_file:
        pickle.dump({"inputs": inputs, "answers": answers}, out_file)

    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
        (train, dev, test),
        batch_size=args.batch_size,
        repeat=False,
        device=args.gpu
    )

    num_tokens = len(inputs.vocab)
    num_labels = len(answers.vocab)

    if args.resume_snapshot:
        model = torch.load(args.resume_snapshot, map_location=lambda storage, locatoin: storage.cuda(args.gpu))
    else:
        model = SNLIClassifier(
            num_tokens,
            num_labels,
            args.embedding_size,
            args.fix_embeddings,
            args.rnn_hidden_size,
            args.rnn_dropout_ratio
        )
        if args.word_vectors:
            model._embedding.weight.data = inputs.vocab.vectors
    model.cuda()

    criterion = nn.CrossEntropyLoss()
    opt = O.Adadelta(model.parameters(), lr=args.learning_rate, weight_decay=args.l2_reg)

    iterations = 0
    stopping_step = 0
    best_dev_acc = -1
    start = time.time()
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
