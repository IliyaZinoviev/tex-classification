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
from os.path import join

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='use cuda or not')
    parser.add_argument('--model', type=str, default='default', help='set name of model dir')
    args = parser.parse_args()
    test(args)


def evaluate_test_set(model, test, x_to_ix, y_to_ix, is_cuda):
    y_true = list()
    y_pred = list()
    for batch, targets, lengths, raw_data in utils.create_dataset(test, x_to_ix, y_to_ix, is_cuda, batch_size=1):
        batch, targets, lengths = utils.sort_batch(batch, targets, lengths)
        pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int())
    if is_cuda:
        y_true = list(map(lambda x: x.cpu(), y_true))
        y_pred = list(map(lambda x: x.cpu(), y_pred))
    print(len(y_true), len(y_pred))
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))


def test(args):
    data_loader = Loader('data/test/')
    assert not args.gpu or (args.gpu and torch.cuda.is_available())
    test_data = data_loader.data
    with open(join('models', args.model, 'params.json'), 'r') as f:
        dicts = json.load(f)
        char_vocab = dicts['chars']
        tag_vocab = dicts['tags']
    model = torch.load(join('models', args.model, 'model.pt'))
    model.eval()
    print('Test samples:', len(test_data))
    print(char_vocab)
    print(tag_vocab)
    test_data = [('$sqrt[8]{x^{8}}$', 'irration_fun'), ('$sqrt[11]{x^{11}}$', 'ration_fun'),
                 ('$sqrt[462]{x^{462}}$', 'irration_fun'), ('$sqrt[1131]{x^{1131}}$', 'ration_fun')]
    evaluate_test_set(model, test_data, char_vocab, tag_vocab, args.gpu)


if __name__ == '__main__':
    main()