# training MEMO-stab
# put this file in the root directory

import random
import argparse
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)

import os
import copy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input

from model.MEMO_stab import *

SEED = 0
#SEED = 100
#SEED = 3000

def set_seeds(seed=SEED):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

def set_global_determinism(seed=SEED):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

set_global_determinism(seed=SEED)

def generate_dataset(dataset):
    maxlen = 512
    input_before_list, input_after_list, input_mask_list, y_list = [], [], [], []
    for i in range(dataset.shape[0]):
        input_before_list.append(np.concatenate([dataset['seq_before_onehot'][i], dataset['seq_before_hhblits'][i]], axis=1))
        input_after_list.append(np.concatenate([dataset['seq_after_onehot'][i], dataset['seq_after_hhblits'][i]], axis=1))
        seq_len = dataset['seq_len'][i]
        mask_array = np.concatenate([np.ones(seq_len,dtype=int),np.zeros(maxlen-seq_len,dtype=int)])
        input_mask_list.append(mask_array)
        y_list.append(dataset['label'][i])

    input_before = np.array(input_before_list)
    input_after = np.array(input_after_list)
    input_mask = np.array(input_mask_list)
    y = np.array(y_list)
    print(input_before.shape)
    print(input_mask.shape)
    print(y.shape)
    
    return [input_before, input_after, input_mask], y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pe','--protein_embedding', default='no')
    parser.add_argument('--max_len', default=512, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.005, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.05, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs") 
    parser.add_argument('--save_dir', default='model/weights/')
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    args = parser.parse_args()

    path = args.save_dir
    if(not os.path.exists(path)):
        os.makedirs(path)

    # load datasets
    train_pkl = pd.read_pickle('datasets/final/train_stab_da(ori)_onehot_hhblits.pkl')
    test_pkl = pd.read_pickle('datasets/final/test_stab_mcsm(all)_onehot_hhblits.pkl')

    # split train/validation
    x_train, y_train = generate_dataset(train_pkl)
    x_test, y_test = generate_dataset(test_pkl)

    batch_size = 32
    protein_embedding = args.protein_embedding
    model = build_MEMO_stab(protein_embedding)
    model.summary()
    #keras.utils.plot_model(model, "model/images/transformer.png", show_shapes=True)
    
    # callbacks
    log = tf.keras.callbacks.CSVLogger(args.save_dir + '/log.csv')
    #tensorboard = tf.keras.callbacks.TensorBoard(log_dir=args.save_dir + model_size + '/tensorboard-logs', histogram_freq=int(args.debug))
    #EarlyStopping = callbacks.EarlyStopping(monitor='val_cc2', min_delta=0.01, patience=5, verbose=0, mode='max', baseline=None, restore_best_weights=True)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(args.save_dir + 'trained_weights.tf', monitor='val_mse', mode='min', #val_categorical_accuracy val_acc
                                       save_best_only=True, save_weights_only=True, verbose=1)        
    #lr_decay = tf.keras.callbacks.LearningRateScheduler(schedule=lambda epoch: args.lr * (args.lr_decay ** epoch))

    # Train the model and save it
    #optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse', 
              optimizer='adam', 
              metrics=['mae', 'mse'])
    
    # no data generator
    
    history = model.fit(x_train, y_train, 
          epochs = args.epochs, verbose=1,
          batch_size = batch_size,
          #validation_split = 0.2,
          validation_data = (x_test, y_test),
          #callbacks = [log, tensorboard, checkpoint, lr_decay],
          callbacks = [log, checkpoint],
          shuffle = True,
          workers = 1).history

    print('Trained model saved to \'%s/trained_model.tf\'' % args.save_dir)