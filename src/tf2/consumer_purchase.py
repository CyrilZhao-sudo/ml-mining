# -*- coding: utf-8 -*-
# Group: MI
# Author: zhao chen
# Date: 2020-09-26

import pandas as pd
import numpy as np
from tensorflow import feature_column
from tensorflow.keras import Model, Input, layers, Sequential
import tensorflow as tf
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

ignore_cols = ["ID"]

train_raw = pd.read_csv("./data/consumer/train_set.csv")
test_raw = pd.read_csv("./data/consumer/test_set.csv")

cate_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
cons_cols = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']

cate_unique_dict = {}
for cate in ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']:
    cate_unique_dict[cate] = train_raw[cate].unique().tolist()


def df_to_dataset(df, shuffle=True, batch_size=32, label_name='label', ignore_cols=[], type=None):
    if not ignore_cols:
        df = df.drop(columns=ignore_cols)
    else:
        df = df.copy()
    if type != 'test':
        labels = df.pop(label_name)
        ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices((dict(df), None))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds

train, valid = train_test_split(train_raw, test_size=0.2)
print(len(train), len(valid), len(test_raw), 'train, valid, test examples')

def build_feature_columns(cate_features, cons_features, cate_unique_dict):
    feature_columns = []
    for cate_col in cate_features:
        column = feature_column.categorical_column_with_vocabulary_list(cate_col, cate_unique_dict[cate_col], default_value=0)
        feature_columns.append(feature_column.indicator_column(column))

    for cons_col in cons_features:
        column = feature_column.numeric_column(cons_col)
        feature_columns.append(column)
    print("feature column: ", feature_columns)
    return feature_columns



feature_columns = build_feature_columns(cate_features=cate_cols, cons_features=cons_cols, cate_unique_dict=cate_unique_dict)

feature_input_layer = layers.DenseFeatures(feature_columns)

batch_size = 32

train_ds = df_to_dataset(train, batch_size=batch_size, ignore_cols=ignore_cols, label_name='y')
val_ds = df_to_dataset(valid, shuffle=False, batch_size=batch_size, ignore_cols=ignore_cols, label_name='y')
test_ds = df_to_dataset(test_raw,shuffle=False, batch_size=batch_size, ignore_cols=ignore_cols, type='test')


model = Sequential([
    feature_input_layer,
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'], run_eagerly=True)

model.fit(train_ds, validation_data=val_ds, epochs=2)


