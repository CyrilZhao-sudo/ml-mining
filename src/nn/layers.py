# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2021/1/29


import torch
import torch.nn.functional as F
import math


class GeneralAttention(torch.nn.Module):
    def __init__(self, input_size, attn_size, p=0.5, **kwargs):
        super(GeneralAttention, self).__init__(**kwargs)
        self.attention = torch.nn.Linear(input_size, attn_size)
        self.projection = torch.nn.Linear(attn_size, 1)
        self.p = p

    def forward(self, x):
        """
        :param x: batch_size, n_filed, dim
        :return:
        """
        if x.dim() != 3:
            raise ValueError('tensor dim is not equal to 3')

        attn_scores = torch.tanh(self.attention(x))
        attn_scores = torch.softmax(self.projection(attn_scores), dim=1)
        attn_scores = torch.dropout(attn_scores, p=self.p, train=self.training)
        attn_output = torch.sum(attn_scores * x, dim=1)
        attn_output = torch.dropout(attn_output, p=self.p, train=self.training)

        return attn_output


class ScaleDotProductAttention(torch.nn.Module):
    def __init__(self, **kwargs):
        super(ScaleDotProductAttention, self).__init__(**kwargs)

    def forward(self, q, k, v, mask=None):
        d = q.size()[-1]
        attn_scores = torch.matmul(q, k.permute(0, 2, 1)) / math.sqrt(d)
        if mask:
            attn_scores = torch.masked_fill(attn_scores, mask==0, -1e9)
        attn_scores = torch.softmax(attn_scores, dim=-1)
        att_output = torch.matmul(attn_scores, v)
        return att_output


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim, n_head, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.n_head = n_head
        self.head_dim = input_dim // n_head
        self.q_w = torch.nn.Linear(input_dim, n_head * self.head_dim)
        self.k_w = torch.nn.Linear(input_dim, n_head * self.head_dim)
        self.v_w = torch.nn.Linear(input_dim, n_head * self.head_dim)
        self.out = torch.nn.Linear(n_head * self.head_dim, n_head * self.head_dim)
        self.attention = ScaleDotProductAttention()

    def forward(self, q, k, v, mask=None):
        batch_size, seq_len, input_dim = q.size()
        q = self.q_w(q).reshape(batch_size, seq_len, self.n_head, self.head_dim).permute(0, 2, 1, 3).reshape(batch_size * self.n_head, seq_len, self.head_dim)
        k = self.k_w(k).reshape(batch_size, seq_len, self.n_head, self.head_dim).permute(0, 2, 1 ,3).reshape(batch_size * self.n_head, seq_len, self.head_dim)
        v = self.v_w(v).reshape(batch_size, seq_len, self.n_head, self.head_dim).permute(0, 2, 1 ,3).reshape(batch_size * self.n_head, seq_len, self.head_dim)
        if mask:
            mask = mask.unsqueeze(1)
        attn_out = self.attention(q, k, v, mask=mask)
        attn_out = attn_out.reshape(batch_size, self.n_head, seq_len, self.head_dim).permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.n_head * self.head_dim)
        out = self.out(attn_out)
        return out


if __name__ == '__main__':
    q = torch.randint(0, 5, (2, 3, 4), dtype=torch.float)
    k = torch.randint(0, 5, (2, 3, 4), dtype=torch.float)
    v = torch.randint(0, 5, (2, 3, 4), dtype=torch.float)
    # att = BahAttention(4, 10)
    # out = att(x)
    # out = ScaleDotProductAttention()(q, k, v)
    out = MultiHeadAttention(input_dim=4, n_head=2)(q, k, v)
    print(out)
    print()
    print(out.shape)
