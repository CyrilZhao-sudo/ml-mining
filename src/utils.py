# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2021/1/8

import numpy as np
import pandas as pd
from datetime import datetime
import random
import torch
import tqdm
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.autograd import Function
from joblib import delayed, Parallel
import itertools

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def setup_display(max_columns=20, max_rows=300, max_colwidth=100):
    # 显示所有列
    pd.set_option('display.max_columns', max_columns)
    # 显示所有行
    pd.set_option('display.max_rows', max_rows)
    # 设置value的显示长度为100，默认为50
    pd.set_option('max_colwidth', max_colwidth)


def get_word_index(seq_list, sep=";"):
    tmp = []
    seq_list = pd.Series(seq_list)
    if sep:
        for seq in seq_list.str.split(sep):
            tmp.extend(list(np.unique(seq)))
    else:
        tmp.extend(list(np.unique(seq_list)))
    word_index = {}
    for target_id in set(tmp):
        word_index[target_id] = len(word_index) + 1
    word_index["unk"] = len(word_index) + 1  # 增加unk
    return word_index


def get_train_test_valid(data, type='train', date_bound=None, date_name="effective_date", label_name='label',
                         seq_name='sequence', test_size=0.2,
                         seed=2020):
    from sklearn.model_selection import train_test_split
    data = data.copy()
    train_test_valid = {}
    if label_name != 'label':
        data['label'] = data[label_name]
    if type == 'train':
        data = data[~pd.isna(data[seq_name])]
        if date_bound is None:
            train, valid = train_test_split(data, test_size=test_size, random_state=seed)
            train_test_valid['train'] = train.reset_index(drop=True)
            train_test_valid['valid'] = valid.reset_index(drop=True)
            train_test_valid['test'] = None
        else:
            train_valid = data[data[date_name] < date_bound]
            train, valid = train_test_split(train_valid, test_size=test_size, random_state=seed)
            test = data[data[date_name] >= date_bound]
            train_test_valid['train'] = train.reset_index(drop=True)
            train_test_valid['valid'] = valid.reset_index(drop=True)
            train_test_valid['test'] = test.reset_index(drop=True)
    else:
        print("data columns: ", ", ".join(data.columns))
        train_test_valid['train'] = None
        train_test_valid['test'] = data.reset_index(drop=True)
        train_test_valid['valid'] = None
    return train_test_valid


class EarlyStopper(object):

    def __init__(self, num_trials, save_path, is_better=True):
        self.num_trials = num_trials
        self.trial_counter = 0
        self.best_metric = 0 if is_better else 1e+10
        self.save_path = save_path
        self.train_loss, self.valid_loss, self.test_loss = None, None, None
        self.train_score, self.valid_score, self.test_score = None, None, None
        self.is_better = is_better
        self.best_epoch = None

    def is_continuable(self, model, metric, losses=None, scores=None, epoch=None):
        if not self.is_better:
            delta_metric = self.best_metric - metric
        else:
            delta_metric = metric - self.best_metric
        if delta_metric > 0:
            self.best_metric = metric
            self.trial_counter = 0
            torch.save(model, self.save_path)
            if losses:
                self.set_losses(losses)
            if scores:
                self.set_scores(scores)
            if epoch:
                self.best_epoch = epoch
            return True
        elif self.trial_counter + 1 < self.num_trials:
            self.trial_counter += 1
            return True
        else:
            return False
    # best valid score's loss
    def set_losses(self, args):
        train_loss, valid_loss, test_loss = args
        self.train_loss = train_loss
        self.valid_loss = valid_loss
        self.test_loss = test_loss
    # best valid score's score
    def set_scores(self, args):
        train_score, valid_score, test_score = args
        self.train_score = train_score
        self.valid_score = valid_score
        self.test_score = test_score
    # get best valid score's loss
    def get_losses(self):
        print(f'{self.best_epoch} valid best losses, train loss {self.train_loss}, valid loss {self.valid_loss}, test loss {self.test_loss}')
        return self.train_loss, self.valid_loss, self.test_loss
    # get best valid score's score
    def get_scores(self):
        print(f'{self.best_epoch} valid best scores, train score {self.train_score}, valid score {self.valid_score}, test score {self.test_score}')
        return self.train_score, self.valid_score, self.test_score



# 单个epoch train
def train_tool(model, optimizer, data_loader, criterion, device, log_interval=0):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (X_batch, y_batch) in enumerate(tk0):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        y_out = model(X_batch)
        # print(y_out.shape)
        loss = criterion(y_out, y_batch.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if log_interval:
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0
    return total_loss / len(data_loader)


def test_tool(model, data_loader, criterion, device, lift=False):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_out = model(X_batch)
            loss = criterion(y_out, y_batch.float())
            y_true.extend(y_batch.tolist())
            y_pred.extend(y_out.tolist())
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    score = roc_auc_score(y_true, y_pred)
    if lift:
        top_p1 = get_lift_value(y_pred, y_true, p=0.01)
        top_p2 = get_lift_value(y_pred, y_true, p=0.02)
        top_p3 = get_lift_value(y_pred, y_true, p=0.03)
        print(f"\t\tlift top1: {round(top_p1[0], 3)} - {top_p1[1]} - {top_p1[2]},"
              f" top2: {round(top_p2[0], 3)} - {top_p2[1]} - {top_p2[2]},"
              f" top3: {round(top_p3[0], 3)} - {top_p3[1]} - {top_p3[2]}")
        return score, avg_loss, max([top_p1[0], top_p2[0], top_p3[0]])
    return score, avg_loss


def predict_prob(model, data_loader, device):
    model.eval()
    y_pred = []
    with torch.no_grad():
        for X_batch, y_batch in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_out = model(X_batch)
            y_pred.extend(y_out.tolist())
    return y_pred


def cross_entropy_loss1d(inputs, targets, cuda=False, scale_pos_weight=1.0):
    """
    :param inputs:
    :param targets:
    :return:
    """
    if cuda:
        targets_array = targets.cpu().numpy()
    else:
        targets_array = targets.numpy()
    pos = (targets_array == 1).sum()
    neg = (targets_array == 0).sum()
    if scale_pos_weight <= 0:
        scale_pos_weight = neg * 1.0 / pos
    # valid = pos + neg
    # weights = np.where(targets_array == 1, neg * 1. / valid, pos * balance / valid)
    weights = np.where(targets_array == 1, scale_pos_weight, 1.0)
    weights = torch.Tensor(weights)
    if cuda:
        weights = weights.cuda()
    loss = nn.BCELoss(weights)(inputs, targets)
    return loss


def get_lift_value(prob, label, p=0.02):
    df = pd.DataFrame({"prob":prob, "label":label})
    df.sort_values(by=['prob'], axis=0, ascending=False, inplace=True, ignore_index=True)
    reject_rank, base_default_rate = int(p * len(df)), np.mean(df["label"])
    prob_threshold = df.loc[reject_rank-1, "prob"]
    reject_df = df.loc[df["prob"]>=prob_threshold, :]
    reject_bad_rate = np.mean(reject_df["label"])
    reject_bad_n = np.sum(reject_df["label"])
    reject_n = len(reject_df)
    return reject_bad_rate / base_default_rate, reject_bad_n, reject_n


class NumpyCrossEntropyLossFunction(Function):
    @staticmethod
    def forward(ctx, inputs, labels):
        # 保存反向传播时需要的数据
        ctx.save_for_backward(inputs.detach(), labels.detach())
        # 转换为numpy类型
        scores = inputs.detach().numpy()
        labels = labels.detach().numpy()
        assert len(scores.shape) == 2
        assert len(labels.shape) == 1
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        N = labels.shape[0]
        correct_probs = probs[range(N), labels] # 花式索引
        loss = -1.0 / N * np.sum(np.log(correct_probs))
        return torch.as_tensor(loss, dtype=inputs.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.detach().numpy()
        inputs, labels = ctx.saved_tensors
        scores = inputs.numpy()
        labels = labels.numpy()

        scores -= np.max(scores, axis=1, keepdims=True)
        exp_score = np.exp(scores)
        probs = exp_score / np.sum(exp_score, axis=1, keepdims=True)
        grad_out = probs
        N = labels.shape[0]
        grad_out[range(N), labels] -= 1
        return torch.from_numpy(grad_out / N), None


class NumpyCrossEntropyLoss(nn.Module):
    def forward(self, inputs, labels):
        return NumpyCrossEntropyLossFunction.apply(inputs, labels)


class FocalLossFunction(Function):
    def __init__(self, alpha, gamma):
        super(FocalLossFunction, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    @staticmethod
    def forward(ctx, inputs, labels):
        pass
    @staticmethod
    def backward(ctx, grad_output):
        pass


class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=0.5, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, target):
        # inputs -> torch.sigmoid(model(x)) (B, 1)
        # target -> (B, 1)
        loss = - (self.alpha * torch.pow(1-inputs, self.gamma) * target * torch.log(inputs) +
                  (1-self.alpha) * torch.pow(inputs, self.gamma) * (1-target) * torch.log(1 - inputs))

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        else:
            assert False
        return loss


def train_tool_with_dynamic_weight(model, optimizer, data_loader, criterion, device, log_interval=0):
    model.train()
    total_loss = 0
    tk0 = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0)
    for i, (X_batch, w_batch, y_batch) in enumerate(tk0):
        X_batch, w_batch, y_batch = X_batch.to(device), w_batch.to(device), y_batch.to(device)
        y_out = model(X_batch)
        loss = criterion(y_out, y_batch.float(), w_batch.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if log_interval:
            if (i+1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0
    return total_loss / len(data_loader)


def wrapper_BECLoss(inputs, targets, weights):
    loss = nn.BCELoss(weights)(inputs, targets)
    return loss


def test_tool_with_dynamic_weight(model, data_loader, criterion, device, lift=False):
    model.eval()
    y_true, y_pred = [], []
    total_loss = 0
    with torch.no_grad():
        for X_batch,w_batch, y_batch in data_loader:
            X_batch, y_batch, w_batch = X_batch.to(device), y_batch.to(device), w_batch.to(device)
            y_out = model(X_batch)
            loss = criterion(y_out, y_batch.float(), w_batch.float())
            y_true.extend(y_batch.tolist())
            y_pred.extend(y_out.tolist())
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    score = roc_auc_score(y_true, y_pred)
    if lift:
        top_p1 = get_lift_value(y_pred, y_true, p=0.01)
        top_p2 = get_lift_value(y_pred, y_true, p=0.02)
        top_p3 = get_lift_value(y_pred, y_true, p=0.03)
        print(f"\tlift top1: {round(top_p1[0], 4)} - {top_p1[1]} - {top_p1[2]},"
              f" top2: {round(top_p2[0], 4)} - {top_p2[1]} - {top_p2[2]},"
              f" top3: {round(top_p3[0], 4)} - {top_p3[1]} - {top_p3[2]}")
        return score, avg_loss, max([top_p1[0], top_p2[0], top_p3[0]])
    return score, avg_loss


"""
计算ks psi
"""


def compute_ks(prob, target):
    '''
    target: numpy array of shape (1,)
    proba: numpy array of shape (1,), predicted probability of the sample being positive
    returns:
    ks: float, ks score estimation
    '''
    from scipy.stats import ks_2samp
    get_ks = lambda prob, target: ks_2samp(prob[target == 1], prob[target != 1]).statistic
    return get_ks(prob, target)


def compute_auc_ks(prob, target, verbose=0):
    from sklearn.metrics import roc_curve, roc_auc_score
    auc = roc_auc_score(y_true=target, y_score=prob)
    fpr, tpr, threshold = roc_curve(y_true=target, y_score=prob)
    ks = max(tpr - fpr)
    if verbose:
        print("\tauc: {0}, ks: {1}".format(round(auc,4), round(ks, 4)))
    return auc, ks


def compute_psi(prob_base, prob_observe, bins=10, verbose=0):
    prob_base = pd.Series(prob_base)
    prob_observe = pd.Series(prob_observe)
    cut_off = np.linspace(0, 1, bins + 1)
    cut_point = sorted(prob_base.quantile(cut_off).unique())
    cut_point[0] = -np.inf
    cut_point[-1] = np.inf
    prob_base_cuts = pd.cut(prob_base, cut_point)
    df1 = pd.DataFrame({'cuts':prob_base_cuts, 'base':prob_base})
    res1 = df1.groupby('cuts')['base'].agg([np.size]).reset_index().rename(columns={'size':'base_size'})

    prob_observe_cuts = pd.cut(prob_observe, cut_point)
    df2 = pd.DataFrame({'cuts':prob_observe_cuts, 'observe':prob_observe})
    res2 = df2.groupby('cuts')['observe'].agg([np.size]).reset_index().rename(columns={'size':'observe_size'})

    res = res1.merge(res2)
    res['base_size'] = res['base_size'] / res['base_size'].sum()
    res['observe_size'] = res['observe_size'] / res['observe_size'].sum()

    psi = np.sum((res['base_size'] - res['observe_size']) * np.log(res['base_size'] / res['observe_size']))
    if verbose:
        print(res, '\n')
        print('psi :', psi)
    return psi


def mapped_id2idx(target_ids, word_index, max_length, sep=';', padding_index='0'):
    unk_idx = word_index.get("unk", '0')
    x_index = [str(word_index.get(t, unk_idx)) for t in target_ids.split(sep)]
    if len(x_index) < max_length:
        x2idx = [padding_index for _ in range(max_length - len(x_index))] + x_index
    else:
        x2idx = x_index[-max_length:]
    return ','.join(x2idx)



def get_lift_value(prob, label, p=0.1):
    df = pd.DataFrame({"prob":prob, "label":label})
    df.sort_values(by=['prob'], axis=0, ascending=False, inplace=True, ignore_index=True)
    reject_rank, base_default_rate = int(p * len(df)), np.mean(df["label"])
    prob_threshold= df.loc[reject_rank-1, "prob"]
    reject_df = df.loc[df["prob"]>=prob_threshold, :]
    reject_bad_rate = np.mean(reject_df["label"])
    reject_bad_n = np.sum(reject_df["label"])
    reject_n = len(reject_df)
    return reject_bad_rate / base_default_rate, reject_bad_n, reject_n

def print_rank_lift(prob, label, percentile=0.2):
    for i in np.arange(0.01, percentile+0.01, 0.01):
        tmp = get_lift_value(prob, label, i)
        print("{}, lift: {}, bad: {}, n: {}".format(round(i, 3), tmp[0], tmp[1], tmp[2]))


def diff_days(x1, x2):
    if np.isnan(x1) or np.isnan(x2):
        return np.nan
    t1 = datetime.strptime(str(int(x1)), "%Y%m%d")
    t2 = datetime.strptime(str(int(x2)), "%Y%m%d")
    diff = (t1 - t2).days
    return diff


class GaussRankScaler:
    def __init__(self):
        self.epsilon = 0.001
        self.lower = -1 + self.epsilon
        self.upper = 1 - self.epsilon
        self.range = self.upper - self.lower
        self.divider = None

    def fit_transform(self, X):
        from scipy.special import erfinv
        i = np.argsort(X, axis=0)
        j = np.argsort(i, axis=0)

        assert (j.min() == 0).all()
        assert (j.max() == len(j) - 1).all()

        j_range = len(j) - 1
        self.divider = j_range / self.range

        transformed = j / self.divider
        transformed = transformed - self.upper
        transformed = erfinv(transformed)

        return transformed


class GetWordIndex:
    def __init__(self, sentences, sep=';', unk_words=None, workers=4, max_vocab_size=None):

        if workers <= 1:
            corpus = self._gen_corpus(sentences, sep=sep, unk_words=unk_words)
        else:
            corpus = Parallel(n_jobs=workers, verbose=1,)(delayed(self._gen_corpus)(_sentences, sep) for _sentences in np.array_split(sentences, workers))
            corpus = itertools.chain.from_iterable(corpus) # iterator

        self.word2idx = self._gen_w2i(corpus, max_vocab_size=max_vocab_size, unk_words=unk_words)

    def _gen_corpus(self, sentences, sep):
        corpus = []
        if sep:
            for sentence in sentences:
                if len(sentence) == 0:
                    continue
                try:
                    word_list = sentence.strip().split(sep)
                    corpus.extend(word_list)
                except:
                    print(sentence)
        else:
            for sentence in sentences:
                if len(sentence) == 0:
                    continue
                corpus.extend(sentence)
        return corpus

    def _gen_w2i(self, corpus, max_vocab_size, unk_words):
        word_index = {}
        if not max_vocab_size:
            for word in set(corpus):
                if unk_words and word in unk_words:
                    continue
                word_index[word] = len(word_index) + 1
        else:
            from collections import Counter
            counter = Counter(corpus)
            for (word, cnt) in counter.most_common(max_vocab_size):
                if unk_words and word in unk_words:
                    continue
                word_index[word] = len(word_index) + 1
        word_index["unk"] = len(word_index) + 1  # 增加unk
        return word_index