import os
import pandas as pd
import numpy as np
import seaborn as sns
import json
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score,recall_score, f1_score,roc_auc_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

protein_embedding_path = '/data/eqtl/memo/protein_embedding_fix/'
config_path = protein_embedding_path + 'summary.config'

# load config
with open(config_path, 'r') as r:
    config_dict = json.load(r)

cutoff = 0.5 
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
    train_pkl1 = pd.read_pickle('/data/eqtl/memo/protein_embedding/train_stab_da(ori)_' + key + '.pkl')
    train_pkl2 = pd.read_pickle('/data/eqtl/memo/protein_embedding/train_stab_da(ori+rev)_' + key + '.pkl')
    #train_pkl3 = pd.read_pickle('../../datasets/final/protein_embedding/train_stab_da(ori+rev+non)_' + key + '.pkl')
    #test_pkl1 = pd.read_pickle('../../datasets/final/protein_embedding/test_stab_da(non)_' + key + '.pkl')
    test_pkl2 = pd.read_pickle('/data/eqtl/memo/protein_embedding/test_stab_da(ori+rev)_' + key + '.pkl')
    test_pkl3 = pd.read_pickle('/data/eqtl/memo/protein_embedding/test_stab_da(rev)_' + key + '.pkl')
    test_pkl4 = pd.read_pickle('/data/eqtl/memo/protein_embedding/test_stab_mcsm(all)_' + key + '.pkl')
    test_pkl5 = pd.read_pickle('/data/eqtl/memo/protein_embedding/test_stab_mcsm(ori)_' + key + '.pkl')
    test_pkl6 = pd.read_pickle('/data/eqtl/memo/protein_embedding/test_stab_mcsm(rev)_' + key + '.pkl')

    shuffled = train_pkl1
    sample_list = []
    y_list = []
    for i in range(shuffled.shape[0]):
        feature_list = []
        feature_list += shuffled['seq_before'][i].flatten().tolist()
        feature_list += shuffled['seq_after'][i].flatten().tolist()
        sample_list.append(feature_list)
        y_list.append(classify(shuffled['label'][i]))
    X_train1 = np.array(sample_list)
    Y_train1 = np.array(y_list)
    #print(X_train1.shape)
    #print(Y_train1.shape)

    shuffled = train_pkl2
    sample_list = []
    y_list = []
    for i in range(shuffled.shape[0]):
        feature_list = []
        feature_list += shuffled['seq_before'][i].flatten().tolist()
        feature_list += shuffled['seq_after'][i].flatten().tolist()
        sample_list.append(feature_list)
        y_list.append(classify(shuffled['label'][i]))
    X_train2 = np.array(sample_list)
    Y_train2 = np.array(y_list)
    #print(X_train2.shape)
    #print(Y_train2.shape)

    #########################################################################

    shuffled = test_pkl2
    sample_list = []
    y_list = []
    for i in range(shuffled.shape[0]):
        feature_list = []
        feature_list += shuffled['seq_before'][i].flatten().tolist()
        feature_list += shuffled['seq_after'][i].flatten().tolist()
        sample_list.append(feature_list)
        y_list.append(classify(shuffled['label'][i]))
    X_test2 = np.array(sample_list)
    Y_test2 = np.array(y_list)

    shuffled = test_pkl3
    sample_list = []
    y_list = []
    for i in range(shuffled.shape[0]):
        feature_list = []
        feature_list += shuffled['seq_before'][i].flatten().tolist()
        feature_list += shuffled['seq_after'][i].flatten().tolist()
        sample_list.append(feature_list)
        y_list.append(classify(shuffled['label'][i]))
    X_test3 = np.array(sample_list)
    Y_test3 = np.array(y_list)

    shuffled = test_pkl4
    sample_list = []
    y_list = []
    for i in range(shuffled.shape[0]):
        feature_list = []
        feature_list += shuffled['seq_before'][i].flatten().tolist()
        feature_list += shuffled['seq_after'][i].flatten().tolist()
        sample_list.append(feature_list)
        y_list.append(classify(shuffled['label'][i]))
    X_test4 = np.array(sample_list)
    Y_test4 = np.array(y_list)

    shuffled = test_pkl5
    sample_list = []
    y_list = []
    for i in range(shuffled.shape[0]):
        feature_list = []
        feature_list += shuffled['seq_before'][i].flatten().tolist()
        feature_list += shuffled['seq_after'][i].flatten().tolist()
        sample_list.append(feature_list)
        y_list.append(classify(shuffled['label'][i]))
    X_test5 = np.array(sample_list)
    Y_test5 = np.array(y_list)

    shuffled = test_pkl6
    sample_list = []
    y_list = []
    for i in range(shuffled.shape[0]):
        feature_list = []
        feature_list += shuffled['seq_before'][i].flatten().tolist()
        feature_list += shuffled['seq_after'][i].flatten().tolist()
        sample_list.append(feature_list)
        y_list.append(classify(shuffled['label'][i]))
    X_test6 = np.array(sample_list)
    Y_test6 = np.array(y_list)

    clf = XGBClassifier(n_estimators=400,learning_rate=0.001)
    clf.fit(X_train1, Y_train1)
    Y_pred2 = clf.predict_proba(X_test2).argmax(axis=1)
    Y_pred3 = clf.predict_proba(X_test3).argmax(axis=1)
    Y_pred4 = clf.predict_proba(X_test4).argmax(axis=1)
    Y_pred5 = clf.predict_proba(X_test5).argmax(axis=1)
    Y_pred6 = clf.predict_proba(X_test6).argmax(axis=1)

    Y_pred2_score = clf.predict_proba(X_test2)
    Y_pred3_score = clf.predict_proba(X_test3)
    Y_pred4_score = clf.predict_proba(X_test4)
    Y_pred5_score = clf.predict_proba(X_test5)
    Y_pred6_score = clf.predict_proba(X_test6)

    acc_list = []
    recall_list = []
    precision_list = []
    f1_list = []
    fpr_list = []
    tpr_list = []
    auc_list = []

    acc_list.append(accuracy_score(Y_test2, Y_pred2))
    acc_list.append(accuracy_score(Y_test3, Y_pred3))
    acc_list.append(accuracy_score(Y_test4, Y_pred4))
    acc_list.append(accuracy_score(Y_pred5, Y_test5))
    acc_list.append(accuracy_score(Y_test6, Y_pred6))

    precision_list.append(precision_score(Y_test2, Y_pred2))
    precision_list.append(precision_score(Y_test3, Y_pred3))
    precision_list.append(precision_score(Y_test4, Y_pred4))
    precision_list.append(precision_score(Y_pred5, Y_test5))
    precision_list.append(precision_score(Y_test6, Y_pred6))

    recall_list.append(recall_score(Y_test2, Y_pred2))
    recall_list.append(recall_score(Y_test3, Y_pred3))
    recall_list.append(recall_score(Y_test4, Y_pred4))
    recall_list.append(recall_score(Y_pred5, Y_test5))
    recall_list.append(recall_score(Y_test6, Y_pred6))

    f1_list.append(f1_score(Y_test2, Y_pred2))
    f1_list.append(f1_score(Y_test3, Y_pred3))
    f1_list.append(f1_score(Y_test4, Y_pred4))
    f1_list.append(f1_score(Y_pred5, Y_test5))
    f1_list.append(f1_score(Y_test6, Y_pred6))

    fpr_list.append(auc_rewrite(to_categorical(Y_test2), Y_pred2_score)[0])
    fpr_list.append(auc_rewrite(to_categorical(Y_test3), Y_pred3_score)[0])
    fpr_list.append(auc_rewrite(to_categorical(Y_test4), Y_pred4_score)[0])
    fpr_list.append(auc_rewrite(to_categorical(Y_test5), Y_pred5_score)[0])
    fpr_list.append(auc_rewrite(to_categorical(Y_test6), Y_pred6_score)[0])

    tpr_list.append(auc_rewrite(to_categorical(Y_test2), Y_pred2_score)[1])
    tpr_list.append(auc_rewrite(to_categorical(Y_test3), Y_pred3_score)[1])
    tpr_list.append(auc_rewrite(to_categorical(Y_test4), Y_pred4_score)[1])
    tpr_list.append(auc_rewrite(to_categorical(Y_test5), Y_pred5_score)[1])
    tpr_list.append(auc_rewrite(to_categorical(Y_test6), Y_pred6_score)[1])

    auc_list.append(auc_rewrite(to_categorical(Y_test2), Y_pred2_score)[2])
    auc_list.append(auc_rewrite(to_categorical(Y_test3), Y_pred3_score)[2])
    auc_list.append(auc_rewrite(to_categorical(Y_test4), Y_pred4_score)[2])
    auc_list.append(auc_rewrite(to_categorical(Y_test5), Y_pred5_score)[2])
    auc_list.append(auc_rewrite(to_categorical(Y_test6), Y_pred6_score)[2])

    fig = plt.figure(figsize=(30,3))

    plt.subplot(1, 3, 3)
    plt.title('test(ori+rev)\nACC=%.2f\nF1=%.2f\nPrecision=%.2f\nRecall=%.2f\nAUC=%.2f'%(acc_list[0],f1_list[0],precision_list[0],recall_list[0],auc_list[0]), fontsize=16,x=0.8,y=0)
    plt.plot(fpr_list[0], tpr_list[0], linewidth = 2,label='AUC=%.3f' % auc_list[0], color=config_dict[key]['color'])
    plt.plot([0,1],[0,1], 'k--')
    plt.axis([0,1.1,0,1.1])
    plt.xlabel('False Postivie Rate')
    plt.ylabel('True Positive Rate')

    plt.subplot(1, 3, 2)
    plt.title('test(rev)\nACC=%.2f\nF1=%.2f\nPrecision=%.2f\nRecall=%.2f\nAUC=%.2f'%(acc_list[1],f1_list[1],precision_list[1],recall_list[1],auc_list[1]), fontsize=16,x=0.8,y=0)
    plt.plot(fpr_list[1], tpr_list[1], linewidth = 2,label='AUC=%.3f' % auc_list[1], color=config_dict[key]['color'])
    plt.plot([0,1],[0,1], 'k--')
    plt.axis([0,1.1,0,1.1])
    plt.xlabel('False Postivie Rate')
    plt.ylabel('True Positive Rate')

    plt.subplot(1, 3, 1)
    plt.title('test(ori)\nACC=%.2f\nF1=%.2f\nPrecision=%.2f\nRecall=%.2f\nAUC=%.2f'%(acc_list[3],f1_list[3],precision_list[3],recall_list[3],auc_list[3]), fontsize=16,x=0.8,y=0)
    plt.plot(fpr_list[3], tpr_list[3], linewidth = 2,label='AUC=%.3f' % auc_list[3], color=config_dict[key]['color'])
    plt.plot([0,1],[0,1], 'k--')
    plt.axis([0,1.1,0,1.1])
    plt.xlabel('False Postivie Rate')
    plt.ylabel('True Positive Rate')

    plt.suptitle('xgboost-' + key + '-training-cutoff' + str(cutoff), fontsize=20)
    plt.savefig('images/' + str(cutoff) + '-xgboost-protein-embedding-noda-training-' + key + '.png',dpi=300, bbox_inches = 'tight')
    #plt.show()


