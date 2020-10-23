# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/10/17

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import regularizers


class LSTMBaseModel(tf.keras.models.Model):
    def __init__(self, max_len, vocab_size, embed_size):
        super(LSTMBaseModel, self).__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.embed_size = embed_size

    def build(self, input_shape):
        self.embedding_layer = tf.keras.layers.Embedding(self.vocab_size, self.embed_size, input_length=self.max_len, embeddings_regularizer=regularizers.l2(0.01))
        self.lstm = tf.keras.layers.LSTM(128, dropout=0.5, recurrent_dropout=0.5, recurrent_regularizer=regularizers.l2(0.01))
        self.dense = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))
        self.output_dense = tf.keras.layers.Dense(1, activation="sigmoid")

        super(LSTMBaseModel, self).build(input_shape)

    def call(self, inputs, training=None, mask=None):
        x = self.embedding_layer(inputs)
        x = self.lstm(x)
        x = self.dense(x)
        out = self.output_dense(x)
        return out

    def init_model(self):
        x_input = tf.keras.layers.Input(shape=(self.max_len,))
        output = self.call(x_input)
        model = tf.keras.Model(inputs=x_input, outputs=output)
        model.summary()
        return model


def create_model(max_len, vocab_size, embed_size):
    tf.keras.backend.clear_session()
    inputs = tf.keras.layers.Input(shape=(max_len,))
    x = tf.keras.layers.Embedding(vocab_size, embed_size, input_length=max_len, embeddings_regularizer=regularizers.l2(0.01))(inputs)
    x = tf.keras.layers.LSTM(128, dropout=0.5, recurrent_dropout=0.5, recurrent_regularizer=regularizers.l2(0.01))(x)
    x = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizers.l2(0.01))(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])
    model.summary()
    return model



MAX_LEN=15
EMBED_SIZE=8
from src.tf2.parpare_data import get_pickle_file
word_index = get_pickle_file("/home/mi/PycharmProjects/BehaviorSequence/resources/word_index")


# tf.keras.backend.clear_session()
# model = LSTMBaseModel(max_len=MAX_LEN, vocab_size=len(word_index)+1, embed_size=10)
# model.build(input_shape=(None, MAX_LEN))
# model.init_model()
# model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["AUC"])

model = create_model(max_len=MAX_LEN, vocab_size=len(word_index)+1, embed_size=EMBED_SIZE)

data_path = "/home/mi/data/sequence/target_ids_b.csv"

def data_csv_generator(data_path=data_path, max_len=MAX_LEN):
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
            yield sequence, label

ds = tf.data.Dataset.from_generator(data_csv_generator, output_types=(tf.int32, tf.int32))

ds = ds.shuffle(buffer_size=10000, seed=123).batch(1024)

model.fit_generator(ds, validation_freq=0.3, epochs=5)



