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

age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65]) # 返回的是one-hot形式
demo(age_buckets)

# 类别型的one-hot
thal = feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'], default_value=0)
thal_one_hot = feature_column.indicator_column(thal)
demo(thal_one_hot)

# embedding
thal_embedding = feature_column.embedding_column(thal, dimension=8)
demo(thal_embedding)

# hash bucket
thal_hashed = feature_column.categorical_column_with_hash_bucket('thal', 100)
demo(feature_column.indicator_column(thal_hashed))

# crossesd
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=100)
demo(feature_column.indicator_column(crossed_feature))


feature_columns = []

# 数值列
for header in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'ca']:
    feature_columns.append(feature_column.numeric_column(header))

# 分桶列
age_buckets = feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])
feature_columns.append(age_buckets)

# 类别类
thal = feature_column.categorical_column_with_vocabulary_list('thal', ['fixed', 'normal', 'reversible'], default_value=0)
thal_one_hot = feature_column.indicator_column(thal)
feature_columns.append(thal_one_hot)

# 嵌入类
thal_embedding = feature_column.embedding_column(thal, dimension=8)
feature_columns.append(thal_embedding)

# 组合列
crossed_feature = feature_column.crossed_column([age_buckets, thal], hash_bucket_size=1000)
crossed_feature = feature_column.indicator_column(crossed_feature)
feature_columns.append(crossed_feature)

feature_layer = keras.layers.DenseFeatures(feature_columns, name="input_feature")


batch_size = 32

train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test,shuffle=False, batch_size=batch_size)


model = keras.Sequential([
    feature_layer,
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc', 'AUC'], run_eagerly=True)


model.fit(train_ds, validation_data=val_ds, epochs=3)


loss, acc = model.evaluate(test_ds)
print("acc", acc)