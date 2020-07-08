# -*- coding: utf-8 -*-
# Group: MI
# Author: zhao chen
# Date: 2020-07-09

'''
    keras 数据生成器
    1.继承Sequence
    2.定义__len__
    3.定义__getitem__
    4.定义on_epoch_end

'''
import keras
import numpy as np

class customDataGenerator(keras.utils.Sequence):

    def __init__(self, X, y, batch_size=24, shuffle=True):
        if not (isinstance(X, np.ndarray) and isinstance(y, np.ndarray)):
            raise Exception("type error. X and y must be ndarray!")
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.X))
        if shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index):
        '''input: batch_indices output: (batch_X, batch_y)'''
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_X, batch_y = self.__generator(batch_indices)

        return batch_X, batch_y

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __generator(self, batch_indices):
        '''get batch_X and batch_y by batch indices'''
        batch_X, batch_y = self.X[batch_indices, :], self.y[batch_indices]

        return np.array(batch_X), np.array(batch_y)

