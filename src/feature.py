# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2021/1/8

import pandas as pd
import numpy as np
from src.utils import get_train_test_valid, setup_seed

setup_seed(2020)


def gen_credit_card_data(data, date_bound=None, date_name=None, seed=2020, output_path=None):
    train_test_valid = get_train_test_valid(data, label_name='Class', seq_name='Class', seed=seed, date_name=date_name,
                                            date_bound=date_bound)
    train, test, valid = train_test_valid['train'], train_test_valid['test'], train_test_valid['valid']
    if date_name is None:
        train.to_csv(output_path + 'credit_card_train.csv', index=False)
        valid.to_csv(output_path + 'credit_card_valid.csv', index=False)
    else:
        train.to_csv(output_path + 'credit_card_train.csv', index=False)
        valid.to_csv(output_path + 'credit_card_valid.csv', index=False)
        test.to_csv(output_path + 'credit_card_test.csv', index=False)


if __name__ == '__main__':
    data = pd.read_csv('~/data/com/creditcard.csv')

    gen_credit_card_data(data, output_path='~/PycharmProjects/ml-mining/data/')
