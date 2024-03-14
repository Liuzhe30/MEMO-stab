# training MEMO-stab
# put this file in the root directory

import random
import argparse
import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import json

import os
import copy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from sklearn.utils import shuffle

from model.MEMO_stab import *

seed_num = 1
if(seed_num == 1):
    SEED = 0
elif(seed_num == 2):
    SEED = 100
elif(seed_num == 3):
    SEED = 3000

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
        input_before_list.append(dataset['seq_before'][i].T)
        input_after_list.append(dataset['seq_after'][i].T)
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
    parser.add_argument('-pe','--protein_embedding', default='yes')
    parser.add_argument('--max_len', default=512, type=int)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--lr', default=0.005, type=float,
                        help="Initial learning rate")
    parser.add_argument('--lr_decay', default=0.05, type=float,
                        help="The value multiplied by lr at each epoch. Set a larger value for larger epochs") 
    parser.add_argument('--save_dir', default='model/weights_pe_da_rev/')
    parser.add_argument('--debug', action='store_true',
                        help="Save weights by TensorBoard")
    args = parser.parse_args()

    path = args.save_dir
    if(not os.path.exists(path)):
        os.makedirs(path)

    protein_embedding_path = 'datasets/protein_embedding_fix/'
    config_path = protein_embedding_path + 'summary.config'

    # load config
    with open(config_path, 'r') as r:
        config_dict = json.load(r)

    for key in config_dict.keys():
        # load datasets
        train_pkl = pd.read_pickle('/data/eqtl/memo/protein_embedding/train_stab_da(ori+rev)_' + key + '.pkl')
        test_pkl2 = pd.read_pickle('/data/eqtl/memo/protein_embedding/test_stab_da(ori+rev)_' + key + '.pkl')
        test_pkl3 = pd.read_pickle('/data/eqtl/memo/protein_embedding/test_stab_da(rev)_' + key + '.pkl')
        test_pkl4 = pd.read_pickle('/data/eqtl/memo/protein_embedding/test_stab_mcsm(all)_' + key + '.pkl')
        test_pkl5 = pd.read_pickle('/data/eqtl/memo/protein_embedding/test_stab_mcsm(ori)_' + key + '.pkl')
        test_pkl6 = pd.read_pickle('/data/eqtl/memo/protein_embedding/test_stab_mcsm(rev)_' + key + '.pkl')
        # split train/validation
        x_train, y_train = generate_dataset(train_pkl)
        x_test2, y_test2 = generate_dataset(test_pkl2)
        x_test3, y_test3 = generate_dataset(test_pkl3)
        x_test4, y_test4 = generate_dataset(test_pkl4)
        x_test5, y_test5 = generate_dataset(test_pkl5)
        x_test6, y_test6 = generate_dataset(test_pkl6)

        batch_size = 16
        protein_embedding = config_dict[key]['embedding_dim']
        model = build_MEMO_stab(protein_embedding)
        model.summary()
        #keras.utils.plot_model(model, "model/images/transformer.png", show_shapes=True)
    
        # callbacks
        log = tf.keras.callbacks.CSVLogger(args.save_dir + '/log.csv')
        #tensorboard = tf.keras.callbacks.TensorBoard(log_dir=args.save_dir + model_size + '/tensorboard-logs', histogram_freq=int(args.debug))
        #EarlyStopping = callbacks.EarlyStopping(monitor='val_cc2', min_delta=0.01, patience=5, verbose=0, mode='max', baseline=None, restore_best_weights=True)
        checkpoint = tf.keras.callbacks.ModelCheckpoint(args.save_dir + 'trained_weights_' + key + '_seed' + str(seed_num) + '.tf', monitor='val_mse', mode='min', #val_categorical_accuracy val_acc
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
            validation_split = 0.2,
            #callbacks = [log, tensorboard, checkpoint, lr_decay],
            callbacks = [log, checkpoint],
            shuffle = False,
            workers = 1).history

        print('Trained model saved to \'%s/trained_model.tf\'' % args.save_dir)

        y_predict2 = model.predict(x_test2)
        y_predict3 = model.predict(x_test3)
        y_predict4 = model.predict(x_test4)
        y_predict5 = model.predict(x_test5)
        y_predict6 = model.predict(x_test6)
        print(y_predict4)
        print(y_test4)
        np.save(path + 'y_pred_stab_da(ori+rev)_' + key + '_seed' + str(seed_num) + '.npy', y_predict2)
        np.save(path + 'y_test_stab_da(ori+rev)_' + key + '.npy', y_test2)
        np.save(path + 'y_pred_stab_da(rev)_' + key + '_seed' + str(seed_num) + '.npy', y_predict3)
        np.save(path + 'y_test_stab_da(rev)_' + key + '.npy', y_test3)
        np.save(path + 'y_pred_stab_mcsm(all)_' + key + '_seed' + str(seed_num) + '.npy', y_predict4)
        np.save(path + 'y_test_stab_mcsm(all)_' + key + '.npy', y_test4)
        np.save(path + 'y_pred_stab_mcsm(ori)_' + key + '_seed' + str(seed_num) + '.npy', y_predict5)
        np.save(path + 'y_test_stab_mcsm(ori)_' + key + '.npy', y_test5)
        np.save(path + 'y_pred_stab_mcsm(rev)_' + key + '_seed' + str(seed_num) + '.npy', y_predict6)
        np.save(path + 'y_test_stab_mcsm(rev)_' + key + '.npy', y_test6)