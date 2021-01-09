# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2021/1/8

import torch
import torch.nn.functional as F

'''
1) 一般的自动编码器
2）使用卷积的自动编码器
3）变分自动编码器

https://www.jeremyjordan.me/autoencoders/

'''


class AutoEncoder(torch.nn.Module):
    def __init__(self, input_dim, hidden_layers, dropout=0.5):
        super(AutoEncoder, self).__init__()
        layers, _input_dim = [], input_dim
        for h_dim in hidden_layers:
            layers.append(torch.nn.Linear(input_dim, h_dim))
            layers.append(torch.nn.BatchNorm1d(h_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = h_dim
        self.encoder = torch.nn.Sequential(*layers)
        layers, de_hidden_layers = [], hidden_layers[:-1][::-1] + [_input_dim]
        for h_dim in de_hidden_layers:
            layers.append(torch.nn.Linear(input_dim, h_dim))
            layers.append(torch.nn.BatchNorm1d(h_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = h_dim
        self.decoder = torch.nn.Sequential(*layers)

    def forward(self, x):
        x_encode = self.encoder(x)
        x_decode = self.decoder(x_encode)
        return x_decode




