# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/9/30

from ..utils import feature_column
from collections import OrderedDict
from tensorflow.keras import Input, layers, Model
from .layers import MultiColumnEmbedding
from .nets import *


class DeepFrameWork:

    def __init__(self, cons_columns=None, cate_columns=None, config=None):
        self.cons_columns = cons_columns  # feature column
        self.cate_columns = cate_columns  # feature column
        self.config = config

    def fit(self, X, y):
        pass

    def build_model(self):
        model = self.__build_model(self.cons_columns, self.cate_columns, config=self.config)
        return model

    def __build_model(self, cons_columns, cate_columns, config):
        '''默认二分类'''
        cons_inputs, cate_inputs = self.__build_inputs(cons_columns, cate_columns)
        cate_embeddings = self.__build_embeddings(cate_columns, cate_inputs, embedding_dropout=0.5)
        cons_dense_layer = self.__build_denses(cons_columns, cons_inputs, dense_dropout=0.5)

        flatten_emb_layer = None
        if len(cate_embeddings) > 0:
            if len(cate_embeddings) == 1:
                flatten_emb_layer = layers.Flatten(name="flatten_cate_embeddings")(cate_embeddings[0])
            else:
                flatten_emb_layer = layers.Flatten(name="flatten_cate_embeddings")(cate_embeddings)

        concat_emb_dense = self.__concat_emb_dense(flatten_emb_layer, cons_dense_layer)
        # DeepFM = ['linear', 'fm_nets', 'dnn_nets']
        linear_output = linear(embeddings=cate_embeddings, flatten_emb_layer=flatten_emb_layer,
                               dense_layer=cons_dense_layer, concat_emb_dense=concat_emb_dense, config=None)
        fm_output = fm_nets(embeddings=cate_embeddings, flatten_emb_layer=flatten_emb_layer,
                               dense_layer=cons_dense_layer, concat_emb_dense=concat_emb_dense, config=None)
        dnn_output = dnn_nets(embeddings=cate_embeddings, flatten_emb_layer=flatten_emb_layer,
                               dense_layer=cons_dense_layer, concat_emb_dense=concat_emb_dense, config=config)
        logit = layers.Add(name="add_logits")([linear_output, fm_output, dnn_output])
        output = layers.Dense(1, activation="sigmoid", name="output", use_bias=True)(logit)

        all_inputs = list(cate_inputs.values()) + list(cons_inputs.values())

        model = Model(inputs=all_inputs, outputs=output)
        model.compile(loss="binary_crossentropy", optimizer="Adam")

        return model


    def __get_feature_column(self, feats, type=None):
        pass

    def __build_inputs(self, cons_columns, cate_columns):
        cons_inputs = OrderedDict()
        cate_inputs = OrderedDict()

        for column in cons_columns:
            cons_inputs[column.name] = Input(shape=(column.input_dim,), name=column.name, dtype=column.dtype)

        cate_inputs["all_cate_feats"] = Input(shape=(len(cate_columns),), name="input_all_cate_feats")

        return cons_inputs, cate_inputs

    def __build_denses(self, cons_columns, cons_inputs, dense_dropout, use_bn=False):
        if cons_inputs:
            if len(cons_inputs) > 1:
                dense_layer = layers.Concatenate(name="concatenate_all_cons_feats")(cons_inputs)
            else:
                dense_layer = list(cons_inputs.values())[0]
        if dense_dropout > 0:
            dense_layer = layers.Dropout(rate=dense_dropout, name="concatenate_all_cons_feats_dropout")(dense_layer)
        if use_bn:
            dense_layer = layers.BatchNormalization(name="concatenate_all_cons_feats_bn")(dense_layer)
        return dense_layer

    def __build_embeddings(self, cate_columns, cate_inputs, embedding_dropout):
        input_layer = cate_inputs["all_cate_feats"]
        input_dims = [column.vocabulary_size for column in cate_columns]
        output_dims = [column.embedding_size for column in cate_columns]
        # TODO 增加embedding的参数
        embeddings = MultiColumnEmbedding(input_dims, output_dims, embedding_dropout,
                                          name="all_cate_embeddings",
                                          embeddings_initializer='uniform',
                                          embeddings_regularizer=None,
                                          activity_regularizer=None,
                                          embeddings_constraint=None,
                                          mask_zero=False)(input_layer)
        return embeddings  # [embedding]

    def __concat_emb_dense(self, flatten_emb_layer, dense_layer):
        if flatten_emb_layer is not None and dense_layer is not None:
            x = layers.Concatenate(name="concat_emb_dense")([flatten_emb_layer, dense_layer])
        elif flatten_emb_layer is not None:
            x = flatten_emb_layer
        elif dense_layer is not None:
            x = dense_layer
        else:
            assert False
        x = layers.BatchNormalization(name="concat_emb_dense_bn")(x)
        return x
