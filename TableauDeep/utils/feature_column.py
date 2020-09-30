# -*- coding: utf-8 -*-

import collections


class CategoricalColumn(collections.namedtuple('CategoricalColumn',
                                               ['name',
                                                'vocabulary_size',
                                                'embedding_size',
                                                'dtype',
                                                'input_name',
                                                ])):
    def __hash__(self):
        return self.name.__hash__()

    def __new__(cls, name, vocabulary_size, embeddings_output_dim=10, dtype='int32', input_name=None, ):
        if input_name is None:
            input_name = '_cat_' + name
        if embeddings_output_dim == 0:
            embeddings_output_dim = int(round(vocabulary_size ** 0.25))
        return super(CategoricalColumn, cls).__new__(cls, name, vocabulary_size, embeddings_output_dim, dtype,
                                                     input_name)


class ContinuousColumn(collections.namedtuple('ContinuousColumn',
                                              ['name',
                                               'column_names',
                                               'input_dim',
                                               'dtype',
                                               'input_name',
                                               ])):
    def __hash__(self):
        return self.name.__hash__()

    def __new__(cls, name, column_names, input_dim=1, dtype='float32', input_name=None, ):
        input_dim = len(column_names)
        return super(ContinuousColumn, cls).__new__(cls, name, column_names, input_dim, dtype, input_name)
