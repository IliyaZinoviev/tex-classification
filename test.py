import os
import sys
import argparse
import time
import random
import utils
import pdb
import json

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from data import PaddedTensorDataset
from data import Loader
from model import LSTMClassifier

def main():
    test()


def apply(model, criterion, batch, targets, lengths):
    pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
    loss = criterion(pred, torch.autograd.Variable(targets))
    return pred, loss

def evaluate_test_set(model, test, x_to_ix, y_to_ix):
    y_true = list()
    y_pred = list()
    for batch, targets, lengths, raw_data in utils.create_dataset(test, x_to_ix, y_to_ix, batch_size=1):
        batch, targets, lengths = utils.sort_batch(batch, targets, lengths)
        pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int())
    y_true = list(map(lambda x: x.cpu(), y_true))
    y_pred = list(map(lambda x: x.cpu(), y_pred))
    print(len(y_true), len(y_pred))
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))


def test():
    data_loader = Loader('data/test/')

    test_data = data_loader.data

    # char_vocab = data_loader.token2id
    # tag_vocab = data_loader.tag2id
    # char_vocab_size = len(char_vocab)
    with open('models/dicts.json', 'r') as f:
        dicts = json.load(f)
        char_vocab = dicts['vocab']
        tag_vocab = dicts['tags']
        char_vocab_size = len(char_vocab)

    print('Test samples:', len(test_data))
    print(char_vocab)
    print(tag_vocab)

    model = torch.load('models/lstm.pt')
    model.eval()
    evaluate_test_set(model, test_data, char_vocab, tag_vocab)


if __name__ == '__main__':
    main()