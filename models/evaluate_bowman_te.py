import random
from argparse import ArgumentParser

import dill as pickle
import torch
from torchtext import data

if __name__ == "__main__":
    random.seed(1235)
    torch.manual_seed(1235)
    parser = ArgumentParser()
    parser.add_argument("--resume_snapshot", type=str, default="")
    parser.add_argument("--test_filename", type=str, required=True)
    parser.add_argument("--index_filename", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    torch.cuda.set_device(args.gpu)

    images = data.Field(sequential=False, unk_token=None, preprocessing=lambda x: x.strip())

    with open(args.index_filename, mode="rb") as in_file:
        index = pickle.load(in_file)
        inputs = index["inputs"]
        answers = index["answers"]
        images = index["images"]

    test = data.TabularDataset(
        path=args.test_filename,
        format="tsv",
        fields=[
            ("label", answers),
            ("premise", inputs),
            ("hypothesis", inputs),
            ("images", images)
        ]
    )

    test_iter = data.BucketIterator(
        test,
        batch_size=args.batch_size,
        repeat=False,
        device=args.gpu
    )

    num_tokens = len(inputs.vocab)
    num_labels = len(answers.vocab)

    if args.resume_snapshot:
        print("Loading model")
        model = torch.load(args.resume_snapshot, map_location=lambda storage, location: storage.cuda(args.gpu))

        print("Evaluating accuracy")
        n_test_correct = 0
        for test_batch_idx, test_batch in enumerate(test_iter):
            print(test_batch_idx, len(test_iter))
            answer = model(test_batch)
            n_test_correct += (
                torch.max(answer, 1)[1].view(test_batch.label.size()).data == test_batch.label.data
            ).sum()
        test_acc = 100. * n_test_correct / len(test)
        print("Test set accuracy: {}".format(test_acc))
