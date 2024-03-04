# generate data augmentated dataset for training
import pandas as pd
import numpy as np
import copy
import sys 
sys.path.append("..") 
from src import fasta2onehot
from src import HHblits2array

fastatool = fasta2onehot.fasta2onehot()
hhmtool = HHblits2array.HHblits2array()

maxlen = 512
output_path = '../datasets/final/'
hhm_path = '../datasets/middlefile/fasta/stab_hhm/'

train_pkl = pd.read_pickle('../datasets/middlefile/train_stab_df.pkl')
test_pkl = pd.read_pickle('../datasets/middlefile/test_stab_df.pkl')
test_reversed_mcsm = pd.read_pickle('../datasets/middlefile/test_stab_reversed_df.pkl')

# ------ preprocess onehot+hhblits ----- 
# generate da training dataset
#print(train_pkl.head())
train_origin_df = pd.DataFrame(columns=['seq_before_onehot', 'seq_after_onehot', 'seq_before_hhblits', 'seq_after_hhblits', 'label'])
train_da_df = pd.DataFrame(columns=['seq_before_onehot', 'seq_after_onehot', 'seq_before_hhblits', 'seq_after_hhblits', 'label'])
train_da_df2 = pd.DataFrame(columns=['seq_before_onehot', 'seq_after_onehot', 'seq_before_hhblits', 'seq_after_hhblits', 'label'])

for i in range(train_pkl.shape[0]):
    seq_before = fastatool.convert(train_pkl['seq_before'][i], maxlen)
    seq_after = fastatool.convert(train_pkl['seq_after'][i], maxlen)
    hhm_before = hhmtool.convert(hhm_path + train_pkl['pdb_id'][i] + '_' + train_pkl['pdb_chain'][i] + '.hhm', maxlen)
    hhm_after = hhmtool.convert(hhm_path + train_pkl['pdb_id'][i] + '_' + train_pkl['pdb_chain'][i] + '_' + train_pkl['shifted_mutation'][i] + '.hhm', maxlen)
    train_origin_df = train_origin_df._append([{'seq_before_onehot':seq_before, 'seq_after_onehot':seq_after, 
                                    'seq_before_hhblits':hhm_before, 'seq_after_hhblits':hhm_after,'label':train_pkl['ddg'][i]}], ignore_index=True)
    train_da_df = train_da_df._append([{'seq_before_onehot':seq_before, 'seq_after_onehot':seq_after, 
                                    'seq_before_hhblits':hhm_before, 'seq_after_hhblits':hhm_after,'label':train_pkl['ddg'][i]}], ignore_index=True)
    train_da_df2 = train_da_df2._append([{'seq_before_onehot':seq_before, 'seq_after_onehot':seq_after, 
                                    'seq_before_hhblits':hhm_before, 'seq_after_hhblits':hhm_after,'label':train_pkl['ddg'][i]}], ignore_index=True)
    train_da_df2 = train_da_df2._append([{'seq_before_onehot':seq_before, 'seq_after_onehot':seq_before, 
                                    'seq_before_hhblits':hhm_before, 'seq_after_hhblits':hhm_before,'label':0}], ignore_index=True)
    

    # reversed items
    seq_before = fastatool.convert(train_pkl['seq_before'][i], maxlen)
    seq_after = fastatool.convert(train_pkl['seq_after'][i], maxlen)
    hhm_before = hhmtool.convert(hhm_path + train_pkl['pdb_id'][i] + '_' + train_pkl['pdb_chain'][i] + '_' + train_pkl['shifted_mutation'][i] + '.hhm', maxlen)
    hhm_after = hhmtool.convert(hhm_path + train_pkl['pdb_id'][i] + '_' + train_pkl['pdb_chain'][i] + '.hhm', maxlen)
    train_da_df = train_da_df._append([{'seq_before_onehot':seq_before, 'seq_after_onehot':seq_after, 
                                    'seq_before_hhblits':hhm_before, 'seq_after_hhblits':hhm_after,'label':-train_pkl['ddg'][i]}], ignore_index=True)
    train_da_df2 = train_da_df2._append([{'seq_before_onehot':seq_before, 'seq_after_onehot':seq_after, 
                                    'seq_before_hhblits':hhm_before, 'seq_after_hhblits':hhm_after,'label':-train_pkl['ddg'][i]}], ignore_index=True)
    train_da_df2 = train_da_df2._append([{'seq_before_onehot':seq_after, 'seq_after_onehot':seq_after, 
                                    'seq_before_hhblits':hhm_after, 'seq_after_hhblits':hhm_after,'label':0}], ignore_index=True)

train_origin_df.to_pickle('../datasets/final/train_stab_da(ori)_onehot_hhblits.pkl')
train_da_df.to_pickle('../datasets/final/train_stab_da(ori+rev)_onehot_hhblits.pkl')
train_da_df2.to_pickle('../datasets/final/train_stab_da(ori+rev+non)_onehot_hhblits.pkl')
print(train_da_df.head())
'''
seq_before_onehot                                   seq_after_onehot                                 seq_before_hhblits                                  seq_after_hhblits  label
0  [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,...  [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...   -0.1
1  [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,...  [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...    0.1
2  [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,...  [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...   -1.8
3  [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,...  [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...    1.8
4  [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,...  [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...   -0.6
'''

# generate reversed mcsm test dataset
test_mcsm_reversed_df = pd.DataFrame(columns=['seq_before_onehot', 'seq_after_onehot', 'seq_before_hhblits', 'seq_after_hhblits', 'label'])
for i in range(test_reversed_mcsm.shape[0]):
    seq_before = fastatool.convert(test_reversed_mcsm['seq_before'][i], maxlen)
    seq_after = fastatool.convert(test_reversed_mcsm['seq_after'][i], maxlen)
    hhm_before = hhmtool.convert(hhm_path + train_pkl['pdb_id'][i] + '_' + train_pkl['pdb_chain'][i] + '_' + train_pkl['shifted_mutation'][i] + '.hhm', maxlen)
    hhm_after = hhmtool.convert(hhm_path + train_pkl['pdb_id'][i] + '_' + train_pkl['pdb_chain'][i] + '.hhm', maxlen)
    test_mcsm_reversed_df = test_mcsm_reversed_df._append([{'seq_before_onehot':seq_before, 'seq_after_onehot':seq_after, 
                                    'seq_before_hhblits':hhm_before, 'seq_after_hhblits':hhm_after,'label':test_reversed_mcsm['ddg'][i]}], ignore_index=True)
test_mcsm_reversed_df.to_pickle('../datasets/final/test_stab_mcsm(rev)_onehot_hhblits.pkl')
print(test_mcsm_reversed_df.head())
'''
                                   seq_before_onehot                                   seq_after_onehot                                 seq_before_hhblits                                  seq_after_hhblits  label
0  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...    1.3
1  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...    1.8
2  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...    1.7
3  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...    0.4
4  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...    1.3
'''

# generate reversed test dataset
test_origin_df = pd.DataFrame(columns=['seq_before_onehot', 'seq_after_onehot', 'seq_before_hhblits', 'seq_after_hhblits', 'label'])
test_da_df = pd.DataFrame(columns=['seq_before_onehot', 'seq_after_onehot', 'seq_before_hhblits', 'seq_after_hhblits', 'label'])
test_reversed_df = pd.DataFrame(columns=['seq_before_onehot', 'seq_after_onehot', 'seq_before_hhblits', 'seq_after_hhblits', 'label'])
test_da_df2 = pd.DataFrame(columns=['seq_before_onehot', 'seq_after_onehot', 'seq_before_hhblits', 'seq_after_hhblits', 'label'])

for i in range(test_pkl.shape[0]):
    seq_before = fastatool.convert(test_pkl['seq_before'][i], maxlen)
    seq_after = fastatool.convert(test_pkl['seq_after'][i], maxlen)
    hhm_before = hhmtool.convert(hhm_path + test_pkl['pdb_id'][i] + '_' + test_pkl['pdb_chain'][i] + '.hhm', maxlen)
    hhm_after = hhmtool.convert(hhm_path + test_pkl['pdb_id'][i] + '_' + test_pkl['pdb_chain'][i] + '_' + test_pkl['shifted_mutation'][i] + '.hhm', maxlen)
    test_origin_df = test_origin_df._append([{'seq_before_onehot':seq_before, 'seq_after_onehot':seq_after, 
                                    'seq_before_hhblits':hhm_before, 'seq_after_hhblits':hhm_after,'label':test_pkl['ddg'][i]}], ignore_index=True)
    test_da_df = test_da_df._append([{'seq_before_onehot':seq_before, 'seq_after_onehot':seq_after, 
                                    'seq_before_hhblits':hhm_before, 'seq_after_hhblits':hhm_after,'label':test_pkl['ddg'][i]}], ignore_index=True)
    test_da_df2 = test_da_df2._append([{'seq_before_onehot':seq_before, 'seq_after_onehot':seq_before, 
                                    'seq_before_hhblits':hhm_before, 'seq_after_hhblits':hhm_before,'label':0}], ignore_index=True)
    # reversed items
    seq_before = fastatool.convert(test_pkl['seq_before'][i], maxlen)
    seq_after = fastatool.convert(test_pkl['seq_after'][i], maxlen)
    hhm_before = hhmtool.convert(hhm_path + test_pkl['pdb_id'][i] + '_' + test_pkl['pdb_chain'][i] + '_' + test_pkl['shifted_mutation'][i] + '.hhm', maxlen)
    hhm_after = hhmtool.convert(hhm_path + test_pkl['pdb_id'][i] + '_' + test_pkl['pdb_chain'][i] + '.hhm', maxlen)
    test_da_df = test_da_df._append([{'seq_before_onehot':seq_before, 'seq_after_onehot':seq_after, 
                                    'seq_before_hhblits':hhm_before, 'seq_after_hhblits':hhm_after,'label':-test_pkl['ddg'][i]}], ignore_index=True)
    test_reversed_df = test_reversed_df._append([{'seq_before_onehot':seq_before, 'seq_after_onehot':seq_after, 
                                    'seq_before_hhblits':hhm_before, 'seq_after_hhblits':hhm_after,'label':-test_pkl['ddg'][i]}], ignore_index=True)
    test_da_df2 = test_da_df2._append([{'seq_before_onehot':seq_after, 'seq_after_onehot':seq_after, 
                                    'seq_before_hhblits':hhm_after, 'seq_after_hhblits':hhm_after,'label':0}], ignore_index=True)

test_origin_df.to_pickle('../datasets/final/test_stab_mcsm(ori)_onehot_hhblits.pkl')
test_da_df.to_pickle('../datasets/final/test_stab_da(ori+rev)_onehot_hhblits.pkl')
test_reversed_df.to_pickle('../datasets/final/test_stab_da(rev)_onehot_hhblits.pkl')
test_da_df2.to_pickle('../datasets/final/test_stab_da(non)_onehot_hhblits.pkl')
print(test_da_df.head())
'''
                                   seq_before_onehot                                   seq_after_onehot                                 seq_before_hhblits                                  seq_after_hhblits  label
0  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...   -1.3
1  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...    1.3
2  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...   -1.8
3  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...    1.8
4  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...  [[99999.0, 99999.0, 99999.0, 99999.0, 99999.0,...   -1.7
'''

# generate mcsm test set
mcsm_test = pd.concat([test_origin_df,test_mcsm_reversed_df])
mcsm_test.to_pickle('../datasets/final/test_stab_mcsm(all)_onehot_hhblits.pkl')