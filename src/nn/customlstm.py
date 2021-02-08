# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/12/17

import torch
from torch import nn
import math


'''

https://towardsdatascience.com/building-a-lstm-by-hand-on-pytorch-59c02a4ec091
http://arunmallya.github.io/writeups/nn/lstm/index.html#/
https://wiseodd.github.io/techblog/2016/08/12/lstm-backprop/

'''

class NaiveCustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        super(NaiveCustomLSTM, self).__init__()
        # forget gate f_t
        self.U_f = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_f = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))

        # input gate i_f
        self.U_i = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_i = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))

        # output gate o_f
        self.U_o = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_o = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))

        # cell c
        self.U_c = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.V_c = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = nn.Parameter(torch.Tensor(hidden_size))

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.init_weights()

    def init_weights(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.uniform_(-stdv, stdv)

    def forward(self, x, init_hidden=None):
        batch_size, seq_length, embed_size = x.shape
        assert embed_size == self.input_size

        if init_hidden is None:
            h_t, c_t = (torch.zeros((batch_size, self.input_size)), torch.zeros((batch_size, self.input_size)))
        else:
            h_t, c_t = init_hidden  # (hidden_size, hidden_size) # h_t_1, c_t_1
        seq_output = []
        for t in range(seq_length):
            x_t = x[:, t, :]
            f_t = torch.sigmoid(x_t @ self.U_f + h_t @ self.V_f + self.b_f)
            i_t = torch.sigmoid(x_t @ self.U_i + h_t @ self.V_i + self.b_i)
            o_t = torch.sigmoid(x_t @ self.U_o + h_t @ self.V_o + self.b_o)
            g_t = torch.tanh(x_t @ self.U_c + h_t @ self.V_c + self.b_c)

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)

            seq_output.append(h_t.unsqueeze(dim=0))
        output = torch.cat(seq_output, dim=0)
        if self.batch_first:
            output = output.permute([1, 0, 2]).contiguous()
        return output, (h_t, c_t)
