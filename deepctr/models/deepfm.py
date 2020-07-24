# -*- coding:utf-8 -*-
"""
Author:
    Weichen Shen,wcshen1994@163.com

Reference:
    [1] Guo H, Tang R, Ye Y, et al. Deepfm: a factorization-machine based neural network for ctr prediction[J]. arXiv preprint arXiv:1703.04247, 2017.(https://arxiv.org/abs/1703.04247)

"""

from itertools import chain
from collections import defaultdict

import tensorflow as tf
from tensorflow.keras import layers

from ..layers.core import PredictionLayer, DNN
from ..layers.interaction import FM
from .base import BaseModel
from ..feature_column import SparseFeat, VarLenSparseFeat, DenseFeat
from ..layers.sequence import SequencePoolingLayer

class DeepFM(BaseModel):
    def __init__(self, linear_feature_columns, dnn_feature_columns,
                 sparse_emb_dim, dnn_layers, dropout_rate=0.5):
        super(DeepFM, self).__init__(linear_feature_columns, dnn_feature_columns,
                                     sparse_emb_dim)

        self.linear = LinearLogitLayer(linear_feature_columns, self.feature_index)
        self.fm = FM()
        # self.dnn= tf.keras.Sequential([
        #                     DNN(sum(map(lambda x: x.dimension, self.dense_feature_columns)) +
        #                           (len(self.sparse_feature_columns) + len(self.varlen_sparse_feature_columns)) * sparse_emb_dim,
        #                           dnn_layers, dropout_rate=dropout_rate),
        #                     layers.Dense(1, use_bias=False, activation='linear')])
        self.dnn = tf.keras.Sequential([
            DNN(hidden_units=dnn_layers, l2_reg=1e-6, dropout_rate=dropout_rate),
            layers.Dense(1, use_bias=False, activation='linear')
        ])

    def call(self, x):
        linear_logit = self.linear(x)

        group_embedding_dict, dense_value_list = self.input_from_feature_columns(x)

        # [[*, 1, embed_size], ]
        sparse_embedding_list = list(chain.from_iterable(group_embedding_dict.values()))
        # [*, embed_size * feat_num]
        sparse_embedding = tf.squeeze(tf.concat(sparse_embedding_list, -1), 1)
        print("HI, see:", tf.shape(sparse_embedding))
        if len(dense_value_list) > 0:
            dnn_input = tf.concat([sparse_embedding, tf.concat(dense_value_list, -1)], axis=-1)
        else:
            dnn_input = sparse_embedding
        dnn_logit = tf.squeeze(self.dnn(dnn_input), 1)

        fm_input = tf.concat(sparse_embedding_list, 1)
        fm_logit = tf.squeeze(self.fm(fm_input), -1)

        pred = linear_logit + dnn_logit + fm_logit + self.out_bias

        return pred


class LinearLogitLayer(layers.Layer):
    def __init__(self, linear_feature_columns, feature_indices, **kwargs) -> None:
        super(LinearLogitLayer, self).__init__(**kwargs)
        self.feature_indices = feature_indices

        self.sparse_feature_columns = list(
                filter(lambda x: isinstance(x, SparseFeat), linear_feature_columns)
                                          ) if len(linear_feature_columns) else []
        self.varlen_sparse_feature_columns = list(
            filter(lambda x: isinstance(x, VarLenSparseFeat), linear_feature_columns)
                                                 ) if linear_feature_columns else []
        self.dense_feature_columns = list(
                filter(lambda x: isinstance(x, DenseFeat), linear_feature_columns)
                                         ) if len(linear_feature_columns) else []
    
        self.embedding_dict = defaultdict(list)
        for feat in self.sparse_feature_columns+self.varlen_sparse_feature_columns:
            seq_mask_zero = False
            prefix = "emb_"
            if isinstance(feat, VarLenSparseFeat):
                seq_mask_zero = True
                prefix = "seq_" + prefix
            emb = layers.Embedding(feat.vocabulary_size, 1,
                            embeddings_initializer='normal',
                            name=prefix + feat.name,
                            mask_zero=seq_mask_zero)
            self.embedding_dict[feat.name] = emb

        if len(self.dense_feature_columns) > 0:
            self.dense_weight = tf.Variable(tf.random.normal(
                                    [sum(fc.dimension for fc in self.dense_feature_columns), 1],
                                    stddev=0.0001), trainable=True)
            
        if len(self.varlen_sparse_feature_columns) > 0:
            self.varlen_pooling_layers = {feat.name: SequencePoolingLayer("sum", supports_masking=True) 
                                            for feat in self.varlen_sparse_feature_columns}
            for feat in self.varlen_sparse_feature_columns:
                print("linear logit init: {}".format(feat.name))

    def call(self, inputs):
        logits = list()
        for fc in self.sparse_feature_columns + self.varlen_sparse_feature_columns:
            feature_name = fc.name

            x = inputs[:, self.feature_indices[feature_name][0]:self.feature_indices[feature_name][1]]
            embed = self.embedding_dict[feature_name](x)

            if isinstance(fc, SparseFeat):
                # [*, 1]
                logits.append(tf.squeeze(embed, [-1]))
            else:
                # [*, 1, 1]
                vec = self.varlen_pooling_layers[feature_name](embed)
                logits.append(tf.squeeze(vec, [-1]))
        # [*, 1]
        logit = tf.math.reduce_sum(tf.concat(logits, -1), axis=1, keepdims=True)

        dense_input_list = []
        for fc in self.dense_feature_columns:
            feature_name = fc.name
            x = inputs[:, self.feature_indices[feature_name][0]:self.feature_indices[feature_name][1]]
            dense_input_list.append(x)
        
        if len(dense_input_list) > 0:
            linear_dense_logit = tf.matmul(tf.concat(dense_input_list, axis=-1), self.weight)
            logit = logit + linear_dense_logit
        
        # [*]
        return tf.squeeze(logit, -1)


# def xDeepFM(linear_feature_columns, dnn_feature_columns, fm_group=[DEFAULT_GROUP_NAME], dnn_hidden_units=(128, 128),
#            l2_reg_linear=0.00001, l2_reg_embedding=0.00001, l2_reg_dnn=0, seed=1024, dnn_dropout=0,
#            dnn_activation='relu', dnn_use_bn=False, task='binary'):
#     """Instantiates the DeepFM Network architecture.

#     :param linear_feature_columns: An iterable containing all the features used by linear part of the model.
#     :param dnn_feature_columns: An iterable containing all the features used by deep part of the model.
#     :param fm_group: list, group_name of features that will be used to do feature interactions.
#     :param dnn_hidden_units: list,list of positive integer or empty list, the layer number and units in each layer of DNN
#     :param l2_reg_linear: float. L2 regularizer strength applied to linear part
#     :param l2_reg_embedding: float. L2 regularizer strength applied to embedding vector
#     :param l2_reg_dnn: float. L2 regularizer strength applied to DNN
#     :param seed: integer ,to use as random seed.
#     :param dnn_dropout: float in [0,1), the probability we will drop out a given DNN coordinate.
#     :param dnn_activation: Activation function to use in DNN
#     :param dnn_use_bn: bool. Whether use BatchNormalization before activation or not in DNN
#     :param task: str, ``"binary"`` for  binary logloss or  ``"regression"`` for regression loss
#     :return: A Keras model instance.
#     """

#     features = build_input_features(
#         linear_feature_columns + dnn_feature_columns)

#     inputs_list = list(features.values())

#     linear_logit = get_linear_logit(features, linear_feature_columns, seed=seed, prefix='linear',
#                                     l2_reg=l2_reg_linear)

#     group_embedding_dict, dense_value_list = input_from_feature_columns(features, dnn_feature_columns, l2_reg_embedding,
#                                                                         seed, support_group=True)

#     fm_logit = add_func([FM()(concat_func(v, axis=-1))
#                          for k, v in group_embedding_dict.items() if k in fm_group])

#     dnn_input = combined_dnn_input(list(chain.from_iterable(
#         group_embedding_dict.values())), dense_value_list)
#     dnn_output = DNN(dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout,
#                      dnn_use_bn, seed)(dnn_input)
#     dnn_logit = tf.keras.layers.Dense(
#         1, use_bias=False, activation=None)(dnn_output)

#     final_logit = add_func([linear_logit, fm_logit, dnn_logit])

#     output = PredictionLayer(task)(final_logit)
#     model = tf.keras.models.Model(inputs=inputs_list, outputs=output)
#     return model
