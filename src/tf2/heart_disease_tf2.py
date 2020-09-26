# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/9/25

import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow import feature_column
from tensorflow import keras
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/heart.csv")

df.head()

train, test = train_test_split(df, test_size=0.2)

train, val = train_test_split(train, test_size=0.2)
print(len(train), 'train examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')


def df_to_dataset(df, shuffle=True, batch_size=32):
    df = df.copy()
    labels = df.pop("target")
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size)
    return ds


batch_size = 5
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_df = df_to_dataset(test, shuffle=False, batch_size=batch_size)

for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print('A batch of ages:', feature_batch['age'])
  print('A batch of targets:', label_batch )

example_batch = next(iter(train_ds))[0]


def demo(feature_column):
    feature_layer = keras.layers.DenseFeatures(feature_column)
    print(feature_layer(example_batch).numpy())


age = feature_column.numeric_column("age")
demo(age)


