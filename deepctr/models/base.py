# -*- coding:utf-8 -*-
"""
Author: Zhaopeng Qiu
Date: create at 2020/7/21

Base model, TF 2.0 API subclassing mode
"""

from collections import defaultdict
from itertools import chain

import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.python.keras.utils.generic_utils import default
from tensorflow.python.ops.gen_math_ops import xdivy

from ..feature_column import build_input_indices, get_linear_logit, DEFAULT_GROUP_NAME, input_from_feature_columns
from ..feature_column import SparseFeat, DenseFeat, VarLenSparseFeat
from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import FM
from ..layers.utils import concat_func, add_func, combined_dnn_input
from ..dense_inputs import embedding_lookup, get_dense_input, varlen_embedding_lookup, get_varlen_pooling_list, mergeDict
from ..dense_inputs import create_embedding_matrix
from ..layers.sequence import SequencePoolingLayer

class BaseModel(tf.keras.Model):
    def __init__(self, linear_feature_columns, dnn_feature_columns, sparse_emb_dim):
        super(BaseModel, self).__init__()

        # 输入的X的位置要和column的顺序保持一致
        # TODO: need improvement
        self.feature_index = build_input_indices(
            linear_feature_columns + dnn_feature_columns)
        self.dnn_feature_columns = dnn_feature_columns

        self.sparse_feature_columns = list(
                filter(lambda x: isinstance(x, SparseFeat), dnn_feature_columns)
                                          ) if len(dnn_feature_columns) else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), dnn_feature_columns)
                                                 ) if dnn_feature_columns else []
        self.dense_feature_columns = list(
                filter(lambda x: isinstance(x, DenseFeat), dnn_feature_columns)
                                         ) if len(dnn_feature_columns) else []

        self.embedding_dict = create_embedding_matrix(dnn_feature_columns, 1e-6)
        # self.embedding_dict = {feat.embedding_name: layers.Embedding(feat.vocabulary_size, sparse_emb_dim, embeddings_initializer='normal')
                                # for feat in self.sparse_feature_columns+self.varlen_sparse_feature_columns}

        self.varlen_pooling_layers = defaultdict(list)
        if len(self.varlen_sparse_feature_columns) > 0:
            self.varlen_pooling_layers = {feat.name: SequencePoolingLayer(feat.combiner, supports_masking=True) 
                                            for feat in self.varlen_sparse_feature_columns}

        self.out_bias= tf.Variable(tf.zeros([1,]), trainable=True)

    def input_from_feature_columns(self, x):
        group_sparse_embedding_dict = embedding_lookup(self.embedding_dict, x, self.sparse_feature_columns, self.feature_index)
        dense_value_list = get_dense_input(x, self.dense_feature_columns, self.feature_index)
        # if not support_dense and len(dense_value_list) > 0:
            # raise ValueError("DenseFeat is not supported in dnn_feature_columns")

        sequence_embed_dict = varlen_embedding_lookup(self.embedding_dict, x, self.varlen_sparse_feature_columns, self.feature_index)
        group_varlen_sparse_embedding_dict = get_varlen_pooling_list(sequence_embed_dict, self.varlen_pooling_layers, x,
                                                                    self.varlen_sparse_feature_columns)
        group_embedding_dict = mergeDict(group_sparse_embedding_dict, group_varlen_sparse_embedding_dict)
        # if not support_group:
            # group_embedding_dict = list(chain.from_iterable(group_embedding_dict.values()))
        return group_embedding_dict, dense_value_list

    def old_input_from_feature_columns(self, x):
        sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            x[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]])
            for feat in self.sparse_feature_columns]
        varlen_sparse_embedding_list = [self.embedding_dict[feat.embedding_name](
            x[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]])
            for feat in self.varlen_sparse_feature_columns]

        dense_value_list = [x[:, self.feature_index[feat.name][0]:self.feature_index[feat.name][1]] for feat in
                            self.dense_feature_columns]

        return sparse_embedding_list, \
               varlen_sparse_embedding_list, \
               dense_value_list
