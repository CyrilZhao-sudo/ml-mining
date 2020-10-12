# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/10/12

import six
from inspect import signature
import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Flatten, BatchNormalization, Activation, Dropout
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from . import layers

def linear(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config):
    """
    Linear(order-1) interactions
    """
    x = None
    x_emb = None
    if embeddings is not None and len(embeddings) > 1:
        concat_embeddings = Concatenate(axis=1, name='concat_linear_embedding')(embeddings)
        x_emb = tf.reduce_sum(concat_embeddings, axis=-1, name='linear_reduce_sum')

    if x_emb is not None and dense_layer is not None:
        x = Concatenate(name='concat_linear_emb_dense')([x_emb, dense_layer])
        # x = BatchNormalization(name='bn_linear_emb_dense')(x)
    elif x_emb is not None:
        x = x_emb
    elif dense_layer is not None:
        x = dense_layer
    else:
        raise ValueError('No input layer exists.')

    x = Dense(1, activation=None, use_bias=False, name='linear_logit')(x)

    return x

def fm_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config):
    """
    FM models pairwise(order-2) feature interactions
    """
    if embeddings is None or len(embeddings) <= 1:
        return None
    fm_concat = Concatenate(axis=1, name='concat_fm_embedding')(embeddings)
    fm_output = layers.FM(name='fm_layer')(fm_concat)
    return fm_output

def dnn_nets(embeddings, flatten_emb_layer, dense_layer, concat_emb_dense, config):
    """
    MLP (fully-connected feed-forward neural nets)
    """
    x_dnn = dnn(concat_emb_dense, config.dnn_params)
    return x_dnn


def dnn(x, params, cellname='dnn'):
    custom_dnn_fn = params.get('custom_dnn_fn')
    if custom_dnn_fn is not None:
        return custom_dnn_fn(x, params, cellname + '_custom')

    hidden_units = params.get('hidden_units', ((128, 0, True), (64, 0, False)))
    activation = params.get('activation', 'relu')
    kernel_initializer = params.get('kernel_initializer', 'he_uniform')
    kernel_regularizer = params.get('kernel_regularizer')
    activity_regularizer = params.get('activity_regularizer')
    if len(hidden_units) <= 0:
        raise ValueError(
            '[hidden_units] must be a list of tuple([units],[dropout_rate],[use_bn]) and at least one tuple.')
    index = 1
    for units, dropout, batch_norm in hidden_units:
        x = Dense(units, use_bias=not batch_norm, name=f'{cellname}_dense_{index}',
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=kernel_regularizer,
                  activity_regularizer=activity_regularizer,
                  )(x)
        if batch_norm:
            x = BatchNormalization(name=f'{cellname}_bn_{index}')(x)
        x = Activation(activation=activation, name=f'{cellname}_activation_{index}')(x)
        if dropout > 0:
            x = Dropout(dropout, name=f'{cellname}_dropout_{index}')(x)
        index += 1
    return x