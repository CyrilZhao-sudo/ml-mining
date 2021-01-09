# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2021/1/8

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import os
import json
import torch
import torchsummary
import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from src.utils import setup_seed, EarlyStopper
from src.dataset import CardFraudDataSet
from src.nn.autoencoders import AutoEncoder

setup_seed(2020)

PATH = "/home/mi/PycharmProjects/ml-mining"
MODEL_PATH = lambda x: PATH + "/resources/ae_files/" + str(x)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def train_ae(model, optimizer, data_loader, criterion, device, log_interval=0):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (X_batch, y_batch) in enumerate(tk0):
        # 无监督学习
        X_batch = X_batch.float().to(device)
        X_out = model(X_batch)
        loss = criterion(X_batch, X_out.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if log_interval:
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0
    return total_loss / len(data_loader)


def test_ae(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.float().to(device)
            X_out = model(X_batch)
            loss = criterion(X_batch, X_out.float())
            total_loss += loss.item()
    return total_loss / len(data_loader)

def get_reconstruct_error(model, data_loader, device):
    def reconstruct_error(X_org, X_new, method='mse'):
        '''MSE'''
        if method == 'mse':
            error = np.mean(np.power(X_org-X_new, 2), axis=1)
        elif method == 'mae':
            error = np.mean(np.abs(X_org-X_new), axis=1)
        else:
            raise ValueError
        return error
    output_X, res_error = []
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            X_out = model(X_batch.float())
            X_batch, X_out = X_batch.numpy(), X_out.numpy()
            error_out = reconstruct_error(X_batch, X_out)
            output_X.append(X_out)
            res_error.append(error_out)
    return np.concatenate(output_X, axis=0), np.concatenate(res_error, axis=0)


def main(args):
    MODE = args.mode
    MODE_TYPE = args.model_type
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs

    NUM_TRIALS = args.num_trials
    INPUT_DIM = args.input_dim
    HIDDEN_LAYERS = eval(args.hidden_layers)
    DROPOUT = args.dropout
    LR = args.lr
    WEIGHT_DECAY = args.weight_decay

    TAG = args.tag if args.tag else datetime.strftime(datetime.now(), "%Y-%m-%d %H:%M:%S")
    MODEL_NAME = '{0}_{1}.pt'.format(TAG, MODE_TYPE)
    MODEL_PARAMS = args.__dict__.copy()

    if MODE == 'train':
        train = pd.read_csv(PATH + '/data/credit_card_train.csv')
        valid = pd.read_csv(PATH + '/data/credit_card_valid.csv')
        train_ds = CardFraudDataSet(train, label_name='Class', feature_names=[f'V{i + 1}' for i in range(28)])
        valid_ds = CardFraudDataSet(valid, label_name='Class', feature_names=[f'V{i + 1}' for i in range(28)])
        train_datalodaer = DataLoader(train_ds, shuffle=True, batch_size=BATCH_SIZE)
        valid_datalodaer = DataLoader(valid_ds, shuffle=True, batch_size=BATCH_SIZE)

        writer = SummaryWriter(PATH + '/logs/ae/{}'.format(TAG.replace(' ', '').replace('-','').replace(':','')))

        model = AutoEncoder(input_dim=INPUT_DIM, hidden_layers=HIDDEN_LAYERS, dropout=DROPOUT)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

        early_stopper = EarlyStopper(num_trials=NUM_TRIALS, save_path=MODEL_PATH(MODEL_NAME), is_better=False)
        best_loss = np.inf
        for epoch in range(EPOCHS):
            train_loss = train_ae(model, optimizer, train_datalodaer, criterion, device=DEVICE)
            valid_loss = test_ae(model, valid_datalodaer, criterion, device=DEVICE)
            print(f'epoch {epoch + 1}, train loss {train_loss}, valid loss {valid_loss}')
            writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss})
            if NUM_TRIALS > 0:
                if not early_stopper.is_continuable(model, valid_loss, losses=(train_loss, valid_loss), epoch=epoch+1):
                    print('early stop. \n')
                    best_valid_losses = early_stopper.get_losses()
                    MODEL_PARAMS.update({'best_losses': best_valid_losses,
                                         'best_epoch': early_stopper.best_epoch})
                    break
            else:
                if valid_loss < best_loss:
                    torch.save(model, MODEL_PATH(MODEL_NAME))
                    MODEL_PARAMS.update({'best_losses': valid_loss,
                                         'best_epoch': epoch+1})

        writer.close()

        with open(MODEL_PATH(TAG + '_params.json'), 'w', encoding='utf8') as f:
            MODEL_PARAMS.update({'model_name': MODEL_NAME, 'tag': TAG})
            json.dump(MODEL_PARAMS, f)

    elif MODE == 'inference':
        INFER_OUTPUT_PATH = args.infer_output_path
        INFER_MODEL_PATH = args.infer_model_path
        INFER_MODEL_TAG = args.infer_model_tag
        INFER_DATA_PATH = args.infer_data_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="track behavior sequence NN model script")
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_trials", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.001)

    parser.add_argument("--dropout", type=float, default=0.5)

    parser.add_argument("--model_type", type=str, default='AE')
    parser.add_argument("--input_dim", type=int, default=28)
    parser.add_argument("--hidden_layers", type=str, default='[48, 16]')
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument('--infer_output_path', type=str, default="/home/mi/PycharmProjects/ml-mining/files/ae")
    parser.add_argument('--infer_model_path', type=str, default="/home/mi/PycharmProjects/ml-mining/resources/ae_files/")
    parser.add_argument('--infer_model_tag', type=str, default=None)
    # TODO 使用新修改的时间差文件打分
    parser.add_argument('--infer_data_path', type=str, default="None")

    args = parser.parse_args()
    main(args)