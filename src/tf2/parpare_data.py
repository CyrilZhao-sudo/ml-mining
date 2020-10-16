# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/10/16

import pandas as pd
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

data_path = "/home/mi/data/sequence/target_ids_b.csv"
word_index_path = "/home/mi/PycharmProjects/BehaviorSequence/resources/word_index"


def get_pickle_file(path):
    import os
    import pickle
    if not os.path.exists(path):
        raise ValueError("the file is not existed!")
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


def save_pickle_file(obj, path):
    import pickle
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


tf.data.Dataset.map()

word_index = get_pickle_file(word_index_path)

data = pd.read_csv(data_path)

tf.keras.backend.clear_session()
ds = tf.data.Dataset.from_tensor_slices((data["sequence"], data["label"]))

max_len = 200


def process_X_y(X, y):
    seq = X.numpy().decode()
    mat = [int(word_index.get(i, 0)) for i in seq.split(";")][:max_len]
    mat = pad_sequences([mat], maxlen=max_len).reshape(-1)
    return mat, int(y)


# ds = ds.map(process_data).repeat(1).batch(2)
ds = ds.map(lambda x, y: tf.py_function(process_X_y, inp=[x, y], Tout=[tf.int32, tf.int32])).repeat(1).batch(5)

for f, l in ds.take(1):
    print(f), print(l)
    break


def process_seq_data(data_path, max_len, word_index):
    pass

def data_csv_generator(data_path="/home/mi/data/sequence/target_ids_b.csv"):
    import csv
    with open(data_path, 'r') as csvfile:
        data = csv.DictReader(csvfile, delimiter=',')
        for _, row in enumerate(data):
            uid, sequence, label = row['uid'], row['sequence'], int(row['label'])
            seq_split = sequence.strip().split(';')
            if len(seq_split) >= max_len:
                sequence = [word_index.get(i, 0) for i in seq_split[-max_len:]]
            else:
                sequence = [0 for _ in range(max_len - len(seq_split))] + [word_index.get(i, 0) for i in seq_split]
            yield uid, sequence, label

ds = tf.data.Dataset.from_generator(data_csv_generator, output_types=(tf.string, tf.int32, tf.int32))





for a,b,c in ds.take(1):
    print(a)
    print(b)
    print(c)
    print(a.numpy().decode())



ds = tf.data.Dataset.from_tensor_slices(([[1,2],[0,3],[-1,3],[-2,3]]))
ds = ds.shuffle(buffer_size=10000).batch(3).repeat(2)
for a in ds:
    print(a)

for epoch in range(epoches):
    for batch in ds:
        pass