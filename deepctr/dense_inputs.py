# -*- coding:utf-8 -*-
"""
Author: Zhaopeng Qiu
Date: create at 2020/7/21

The input is a dense vector
"""

from collections import defaultdict
from itertools import chain

from tensorflow.python.keras.layers import Embedding
from tensorflow.python.keras.regularizers import l2

# from . import feature_column as fc_lib
from .layers.sequence import SequencePoolingLayer, WeightedSequenceLayer
from .layers.utils import Hash


def get_inputs_list(inputs):
    return list(chain(*list(map(lambda x: x.values(), filter(lambda x: x is not None, inputs)))))


def create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, l2_reg,
                          prefix='sparse_', seq_mask_zero=True):
    sparse_embedding = {}
    for feat in sparse_feature_columns:
        emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                        embeddings_initializer=feat.embeddings_initializer,
                        embeddings_regularizer=l2(l2_reg),
                        name=prefix + '_emb_' + feat.embedding_name)
        emb.trainable = feat.trainable
        sparse_embedding[feat.embedding_name] = emb

    if varlen_sparse_feature_columns and len(varlen_sparse_feature_columns) > 0:
        for feat in varlen_sparse_feature_columns:
            # if feat.name not in sparse_embedding:
            emb = Embedding(feat.vocabulary_size, feat.embedding_dim,
                            embeddings_initializer=feat.embeddings_initializer,
                            embeddings_regularizer=l2(
                                l2_reg),
                            name=prefix + '_seq_emb_' + feat.name,
                            mask_zero=seq_mask_zero)
            emb.trainable = feat.trainable
            sparse_embedding[feat.embedding_name] = emb
    return sparse_embedding


def get_embedding_vec_list(embedding_dict, input_dict, sparse_feature_columns, return_feat_list=(), mask_feat_list=()):
    embedding_vec_list = []
    for fg in sparse_feature_columns:
        feat_name = fg.name
        if len(return_feat_list) == 0 or feat_name in return_feat_list:
            if fg.use_hash:
                lookup_idx = Hash(fg.vocabulary_size, mask_zero=(feat_name in mask_feat_list))(input_dict[feat_name])
            else:
                lookup_idx = input_dict[feat_name]

            embedding_vec_list.append(embedding_dict[feat_name](lookup_idx))

    return embedding_vec_list


def create_embedding_matrix(feature_columns, l2_reg, prefix="", seq_mask_zero=True):
    sparse_feature_columns = list(
        filter(lambda x: type(x).__name__ == "SparseFeat", feature_columns)) if feature_columns else []
    varlen_sparse_feature_columns = list(
        filter(lambda x: type(x).__name__ == "VarLenSparseFeat", feature_columns)) if feature_columns else []
    sparse_emb_dict = create_embedding_dict(sparse_feature_columns, varlen_sparse_feature_columns, 
                                            l2_reg, prefix=prefix + 'sparse', seq_mask_zero=seq_mask_zero)
    return sparse_emb_dict


def embedding_lookup(sparse_embedding_dict, dense_inputs, sparse_feature_columns, feature_index, return_feat_list=(),
                     mask_feat_list=(), to_list=False):
    group_embedding_dict = defaultdict(list)
    for fc in sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name

        x = dense_inputs[:, feature_index[feature_name][0]:feature_index[feature_name][1]]
        if (len(return_feat_list) == 0 or feature_name in return_feat_list):
            if fc.use_hash:
                lookup_idx = Hash(fc.vocabulary_size, mask_zero=(feature_name in mask_feat_list))(x)
            else:
                lookup_idx = x

            group_embedding_dict[fc.group_name].append(sparse_embedding_dict[embedding_name](lookup_idx))
    if to_list:
        return list(chain.from_iterable(group_embedding_dict.values()))
    return group_embedding_dict


def varlen_embedding_lookup(embedding_dict, dense_inputs, varlen_sparse_feature_columns, feature_index):
    """
    embedding_dict: dict[featurename, EmbeddingMatrix]
    dense_inputs: shape [batch_size, raw_vector_size]
    feature_index: dict[featurename, (start_index, end_index)]
    """
    varlen_embedding_vec_dict = {}
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        embedding_name = fc.embedding_name

        x = dense_inputs[:, feature_index[feature_name][0]:feature_index[feature_name][1]]
        if fc.use_hash:
            lookup_idx = Hash(fc.vocabulary_size, mask_zero=True)(x)
        else:
            lookup_idx = x
        varlen_embedding_vec_dict[feature_name] = embedding_dict[embedding_name](lookup_idx)
    return varlen_embedding_vec_dict


def get_varlen_pooling_list(embedding_dict, pooling_layer_dict, features, varlen_sparse_feature_columns, to_list=False):
    pooling_vec_list = defaultdict(list)
    for fc in varlen_sparse_feature_columns:
        feature_name = fc.name
        seq_input = embedding_dict[feature_name]
        vec = pooling_layer_dict[feature_name](seq_input)
        pooling_vec_list[fc.group_name].append(vec)
    if to_list:
        return chain.from_iterable(pooling_vec_list.values())
    return pooling_vec_list


def get_dense_input(dense_inputs, dense_feature_columns, feature_index):
    dense_feature_columns = list(
        filter(lambda x: type(x).__name__ == "DenseFeat", dense_feature_columns)) if dense_feature_columns else []
    dense_input_list = []
    for fc in dense_feature_columns:
        feature_name = fc.name

        x = dense_inputs[:, feature_index[feature_name][0]:feature_index[feature_name][1]]
        dense_input_list.append(x)
    return dense_input_list


def mergeDict(a, b):
    c = defaultdict(list)
    for k, v in a.items():
        c[k].extend(v)
    for k, v in b.items():
        c[k].extend(v)
    return c
