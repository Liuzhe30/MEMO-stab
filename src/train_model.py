import os
import pandas as pd
import numpy as np
import seaborn as sns
import json
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,recall_score, f1_score,roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import pickle

protein_embedding_path = 'datasets/protein_embedding_fix/'
config_path = protein_embedding_path + 'summary.config'

# load config
with open(config_path, 'r') as r:
    config_dict = json.load(r)

n_estimators = 1500
cutoff = 0.4
def classify(label):
    if(np.abs(label) >= cutoff):
        label = 1
    else:
        label = 0
    return label

def auc_rewrite(y_one_hot, y_score_pro):
    fpr, tpr, thresholds = roc_curve(y_one_hot.ravel(),y_score_pro.ravel())   
    return fpr, tpr, auc(fpr, tpr)

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import r2_score, mean_squared_error
from scipy.stats import pearsonr
from sklearn.model_selection import cross_val_score, KFold
sns.set(color_codes=True)

#key_list = list(config_dict.keys())
for key in config_dict.keys():
    print(key)
    # prepare datasets
    train_pkl = pd.read_pickle('datasets/final/protein_embedding/train_stab_da(ori)_' + key + '.pkl')
    test_pkl = pd.read_pickle('datasets/final/protein_embedding/test_stab_mcsm(ori)_' + key + '.pkl')

    shuffled = train_pkl
    sample_list = []
    y_list = []
    for i in range(shuffled.shape[0]):
        feature_list = []
        feature_list += shuffled['seq_before'][i].flatten().tolist()
        feature_list += shuffled['seq_after'][i].flatten().tolist()
        sample_list.append(feature_list)
        y_list.append(classify(shuffled['label'][i]))
    X_train = np.array(sample_list)
    Y_train = np.array(y_list)

    #########################################################################

    shuffled = test_pkl
    sample_list = []
    y_list = []
    for i in range(shuffled.shape[0]):
        feature_list = []
        feature_list += shuffled['seq_before'][i].flatten().tolist()
        feature_list += shuffled['seq_after'][i].flatten().tolist()
        sample_list.append(feature_list)
        y_list.append(classify(shuffled['label'][i]))
    X_test = np.array(sample_list)
    Y_test = np.array(y_list)

    clf = XGBClassifier(n_estimators=n_estimators,learning_rate=0.001)
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict_proba(X_test).argmax(axis=1)
    Y_pred_score = clf.predict_proba(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    precision = precision_score(Y_test, Y_pred)
    recall = recall_score(Y_test, Y_pred)
    f1 = f1_score(Y_test, Y_pred)

    fpr = auc_rewrite(to_categorical(Y_test), Y_pred_score)[0]
    tpr = auc_rewrite(to_categorical(Y_test), Y_pred_score)[1]
    auc_ = auc_rewrite(to_categorical(Y_test), Y_pred_score)[2]

    fig = plt.figure(figsize=(4,4))

    plt.title('ACC=%.2f\nF1=%.2f\nPrecision=%.2f\nRecall=%.2f\nAUC=%.2f'%(acc,f1,precision,recall,auc_), fontsize=16,x=0.7,y=0)
    plt.plot(fpr, tpr, linewidth = 5,label='AUC=%.3f' % auc_, color=config_dict[key]['color'])
    plt.plot([0,1],[0,1], 'k--')
    plt.axis([0,1.1,0,1.1])
    plt.xlabel('False Postivie Rate')
    plt.ylabel('True Positive Rate')

    #plt.suptitle('xgboost-' + key + '-training-nestimators' + str(n_estimators), fontsize=20)
    plt.suptitle('xgboost-' + key, fontsize=20)
    plt.savefig('images_sin1500/' + str(cutoff) + '-xgboost-protein-embedding-' + key + '.png',dpi=300, bbox_inches = 'tight')
    #plt.show()


