# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2021/1/8

import numpy as np
from torch.utils.data import Dataset


class CardFraudDataSet(Dataset):

    def __init__(self, data, label_name, feature_names):
        self.data = data
        self.X = data[feature_names]
        self.y = data[label_name]

    def __getitem__(self, item):
        x = self.X.loc[item]
        y = self.y.loc[item]

        return np.array(x), np.array(y)

    def __len__(self):
        return len(self.data)

