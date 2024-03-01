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

def build_MEMO_stab():

    # hyper-paramaters
    maxlen = 512
    block_size = 20
    embed_dim = 64
    num_heads = 4
    ff_dim = 256