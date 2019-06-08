import os
import sys
import argparse
import time
import random
import utils
import pdb

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from os import makedirs
from os.path import join, exists
from json import dump

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from data import PaddedTensorDataset
from data import TrainLoader
from model import LSTMClassifier
from test import evaluate_test_set


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data',
                        help='data_directory')
    parser.add_argument('--hidden_dim', type=int, default=32,
                        help='LSTM hidden dimensions')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='size for each minibatch')
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='maximum number of epochs')
    parser.add_argument('--char_dim', type=int, default=64,
                        help='character embedding dimensions')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight_decay rate')
    parser.add_argument('--seed', type=int, default=123,
                        help='seed for random initialisation')
    parser.add_argument('--gpu', type=int, default=0,
                        help='use cuda or not')
    parser.add_argument('--save_to', type=str, default='default',
                        help='set dir name')
    args = parser.parse_args()
    train(args)


def apply(model, criterion, batch, targets, lengths):
    pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
    loss = criterion(pred, torch.autograd.Variable(targets))
    return pred, loss


def train_model(model, optimizer, train, dev, x_to_ix, y_to_ix, batch_size, max_epochs, is_cuda):
    criterion = nn.NLLLoss(size_average=False)
    for epoch in range(max_epochs):
        print('Epoch:', epoch)
        y_true = list()
        y_pred = list()
        total_loss = 0
        for batch, targets, lengths, raw_data in utils.create_dataset(train, x_to_ix, y_to_ix, is_cuda,
                                                                      batch_size=batch_size):
            batch, targets, lengths = utils.sort_batch(batch, targets, lengths)
            model.zero_grad()
            pred, loss = apply(model, criterion, batch, targets, lengths)
            loss.backward()
            optimizer.step()

            pred_idx = torch.max(pred, 1)[1]
            y_true += list(targets.int())
            y_pred += list(pred_idx.data.int())
            total_loss += loss
            if is_cuda:
                y_true = list(map(lambda x: x.cpu(), y_true))
                y_pred = list(map(lambda x: x.cpu(), y_pred))
        acc = accuracy_score(y_true, y_pred)
        val_loss, val_acc = evaluate_validation_set(model, dev, x_to_ix, y_to_ix, criterion, is_cuda)
        print(
            "Train loss: {} - acc: {} \nValidation loss: {} - acc: {}".format(total_loss.data.float() / len(train), acc,
                                                                              val_loss, val_acc))
    return model


def evaluate_validation_set(model, devset, x_to_ix, y_to_ix, criterion, is_cuda):
    y_true = list()
    y_pred = list()
    total_loss = 0
    for batch, targets, lengths, raw_data in utils.create_dataset(devset, x_to_ix, y_to_ix, is_cuda, batch_size=1):
        batch, targets, lengths = utils.sort_batch(batch, targets, lengths)
        pred, loss = apply(model, criterion, batch, targets, lengths)
        pred_idx = torch.max(pred, 1)[1]
        y_true += list(targets.int())
        y_pred += list(pred_idx.data.int())
        total_loss += loss
    if is_cuda:
        y_true = list(map(lambda x: x.cpu(), y_true))
        y_pred = list(map(lambda x: x.cpu(), y_pred))
    acc = accuracy_score(y_true, y_pred)
    return total_loss.data.float() / len(devset), acc


def save_model(model, params, dir):
    if not exists(join('models', dir)):
        makedirs(join('models', dir))
    torch.save(model, join('models', dir, 'model.pt'))
    with open(join('models', dir, 'params.json'), 'w') as outfile:
        dump(params, outfile)


def train(args):
    print(args)
    assert not args.gpu or (args.gpu and torch.cuda.is_available())
    random.seed(args.seed)
    data_loader = TrainLoader(args.data_dir)

    train_data = data_loader.train_data
    dev_data = data_loader.dev_data
    test_data = data_loader.test_data

    char_vocab = data_loader.token2id
    tag_vocab = data_loader.tag2id
    char_vocab_size = len(char_vocab)

    print('Training samples:', len(train_data))
    print('Valid samples:', len(dev_data))
    print('Test samples:', len(test_data))

    print(char_vocab)
    print(tag_vocab)

    model = LSTMClassifier(char_vocab_size, args.char_dim, args.hidden_dim, len(tag_vocab), args.gpu)
    if args.gpu:
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    model = train_model(model, optimizer, train_data, dev_data, char_vocab, tag_vocab, args.batch_size, args.num_epochs,
                        args.gpu)

    save_model(model, {'chars': char_vocab, 'tags': tag_vocab}, args.save_to)
    evaluate_test_set(model, test_data, char_vocab, tag_vocab, args.gpu)


if __name__ == '__main__':
    main()