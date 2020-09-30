# -*- coding: utf-8 -*-
# @Author : Cyril
# @Date   : 2019/10/18
# @File   : util_functions.py

'''loss function'''
import numpy as np

def squared_hinge_loss(preds, train_data):
    '''squared hinge loss'''
    labels = train_data.get_label()
    grad = 2 * labels * labels * preds - 2 * labels
    grad = np.where(preds * labels >= 1, 0, grad)
    hess = 2 * labels * labels
    hess = np.where(preds * labels >= 1 , 0, hess)
    return grad, hess

def log_likelihood_loss(preds, train_data):
     labels = train_data.get_label()
     preds = 1. / (1. + np.exp(-preds))
     grad = preds - labels
     hess = preds * (1. - preds)
     return grad, hess


def huber_approx_obj(preds, dtrain):
    d = preds - dtrain.get_label()
    h = 1  # h is delta in the graphic
    scale = 1 + (d / h) ** 2
    scale_sqrt = np.sqrt(scale)
    grad = d / scale_sqrt
    hess = 1 / scale / scale_sqrt
    return grad, hess


def fair_obj(preds, dtrain):
    """y = c * abs(x) - c**2 * np.log(abs(x)/c + 1)"""
    x = preds - dtrain.get_label()
    c = 1
    den = abs(x) + c
    grad = c * x / den
    hess = c * c / den ** 2
    return grad, hess


def log_cosh_obj(preds, dtrain):
    x = preds - dtrain.get_label()
    grad = np.tanh(x)
    hess = 1 / np.cosh(x) ** 2
    return grad, hess


def grad(preds, dtrain):
    labels = dtrain.get_label()
    n = preds.shape[0]
    grad = np.empty(n)
    hess = 500 * np.ones(n)
    for i in range(n):
        diff = preds[i] - labels[i]
        if diff > 0:
            grad[i] = 200
        elif diff < 0:
            grad[i] = -200
        else:
            grad[i] = 0
    return grad, hess


'''自定义评价函数 feval'''
# another self-defined eval metric
# f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
# accuracy
def accuracy(preds, train_data):
    labels = train_data.get_label()
    return 'accuracy', np.mean(labels == (preds > 0.5)), True

# self-defined eval metric
# f(preds: array, train_data: Dataset) -> name: string, eval_result: float, is_higher_better: bool
# binary error
def binary_error(preds, train_data):
    labels = train_data.get_label()
    return 'error', np.mean(labels != (preds > 0.5)), False



def robust_pow(num_base, num_pow):
    # numpy does not permit negative numbers to fractional power
    # use this to perform the power algorithmic
    return np.sign(num_base) * (np.abs(num_base)) ** (num_pow)

def focal_binary_object(pred, dtrain):
    gamma_indct = 2.5
    # retrieve data from dtrain matrix
    label = dtrain.get_label()
    # compute the prediction with sigmoid
    sigmoid_pred = 1.0 / (1.0 + np.exp(-pred))
    # gradient
    # complex gradient with different parts
    g1 = sigmoid_pred * (1 - sigmoid_pred)
    g2 = label + ((-1) ** label) * sigmoid_pred
    g3 = sigmoid_pred + label - 1
    g4 = 1 - label - ((-1) ** label) * sigmoid_pred
    g5 = label + ((-1) ** label) * sigmoid_pred
    # combine the gradient
    grad = gamma_indct * g3 * robust_pow(g2, gamma_indct) * np.log(g4 + 1e-9) + \
           ((-1) ** label) * robust_pow(g5, (gamma_indct + 1))
    # combine the gradient parts to get hessian components
    hess_1 = robust_pow(g2, gamma_indct) + \
             gamma_indct * ((-1) ** label) * g3 * robust_pow(g2, (gamma_indct - 1))
    hess_2 = ((-1) ** label) * g3 * robust_pow(g2, gamma_indct) / g4
    # get the final 2nd order derivative
    hess = ((hess_1 * np.log(g4 + 1e-9) - hess_2) * gamma_indct +
            (gamma_indct + 1) * robust_pow(g5, gamma_indct)) * g1

    return grad, hess