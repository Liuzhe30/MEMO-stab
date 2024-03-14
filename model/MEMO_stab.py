# model structure 

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from .utils.Transformer import MultiHeadSelfAttention
from .utils.Transformer import TransformerBlock
from .utils.Transformer import TokenAndPositionEmbedding

def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    print(seq.shape)
    # add extra dimensions to add the padding
    # to the attention logits.
    return  seq[:, tf.newaxis, tf.newaxis, :]# (batch_size, 1, 1, seq_len)

def build_MEMO_stab(protein_embedding):

    # hyper-paramaters
    vocab_size = 5
    maxlen = 512
    block_size = 20
    embed_dim = 64
    num_heads = 4
    ff_dim = 256
    pos_embed_dim = 64

    if(protein_embedding == 'no'):
        
        seq_embed_dim_bet = 14
        input1 = Input(shape=(maxlen, 20+30), name = 'input_before') 
        input2 = Input(shape=(maxlen, 20+30), name = 'input_after') 
        input3 = Input(shape=(maxlen,), name = 'input_mask') 
    else:
        embed_dim = 1288
        pos_embed_dim = 1288
        seq_embed_dim_bet = pos_embed_dim - protein_embedding
        input1 = Input(shape=(maxlen, protein_embedding), name = 'input_before') 
        input2 = Input(shape=(maxlen,protein_embedding), name = 'input_after') 
        input3 = Input(shape=(maxlen,), name = 'input_mask') 

    before_mask = create_padding_mask(input3)
    after_mask = create_padding_mask(input3)

    # -----init all used basic layers-----
    embedding_layer_before = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, pos_embed_dim, seq_embed_dim_bet)
    embedding_layer_after = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim, pos_embed_dim, seq_embed_dim_bet)

    # before branch
    trans_block_before1 = TransformerBlock(embed_dim, num_heads, ff_dim)
    trans_block_before2 = TransformerBlock(embed_dim, num_heads, ff_dim)
    trans_block_before3 = TransformerBlock(embed_dim, num_heads, ff_dim)

    # after branch
    trans_block_after1 = TransformerBlock(embed_dim, num_heads, ff_dim)
    trans_block_after2 = TransformerBlock(embed_dim, num_heads, ff_dim)
    trans_block_after3 = TransformerBlock(embed_dim, num_heads, ff_dim)

    # -----generate-----
    print("generate")

    before = embedding_layer_before([input3,input1])
    after = embedding_layer_after([input3,input2])
    print('embedding_layer_before.get_shape()', before.get_shape())
    print('embedding_layer_after.get_shape()', after.get_shape()) 

    before = trans_block_before1(before, before_mask)
    before = trans_block_before2(before, before_mask)
    before = trans_block_before3(before, before_mask)
    after = trans_block_after1(after, after_mask)
    after = trans_block_after2(after, after_mask)
    after = trans_block_after3(after, after_mask)

    # concatenate
    merged = tf.keras.layers.Concatenate(axis=1)([before, after])
    print('merged.get_shape()', merged.get_shape()) 
    merged = layers.Dense(128)(merged)
    merged = layers.Dense(32)(merged)
    merged = layers.Flatten()(merged)
    merged = layers.Dense(32)(merged)
    output = layers.Dense(1)(merged)

    model = Model(inputs=[input1, input2, input3], outputs=output)
    return model


