# generate protein embedding dataset for training
import pandas as pd
import numpy as np
import json

protein_embedding_path = '../datasets/protein_embedding_fix/'
config_path = protein_embedding_path + 'summary.config'

# load config
with open(config_path, 'r') as r:
    config_dict = json.load(r)

maxlen = 512
output_path = '../datasets/final/'

train_pkl = pd.read_pickle('../datasets/middlefile/train_stab_df.pkl')
test_pkl = pd.read_pickle('../datasets/middlefile/test_stab_df.pkl')
test_reversed_mcsm = pd.read_pickle('../datasets/middlefile/test_stab_reversed_df.pkl')

# ------ preprocess protein embedding ----- 
# generate da training dataset
print(train_pkl.head())
'''
  pdb_id pdb_chain uniprot_id mutation shifted_mutation  ddg                                         seq_before                                          seq_after
0   1PY6         A      space      E9A              E5A -0.1  TGRPEWIWLALGTALMGLGTLYFLVKGMGVSDPDAKKFYAITTLVP...  TGRPAWIWLALGTALMGLGTLYFLVKGMGVSDPDAKKFYAITTLVP...
1   1PY6         A      space     L13A              L9A -1.8  TGRPEWIWLALGTALMGLGTLYFLVKGMGVSDPDAKKFYAITTLVP...  TGRPEWIWAALGTALMGLGTLYFLVKGMGVSDPDAKKFYAITTLVP...
2   1PY6         A      space     A39P             A35P -0.6  TGRPEWIWLALGTALMGLGTLYFLVKGMGVSDPDAKKFYAITTLVP...  TGRPEWIWLALGTALMGLGTLYFLVKGMGVSDPDPKKFYAITTLVP...
3   1PY6         A      space     F42A             F38A -2.0  TGRPEWIWLALGTALMGLGTLYFLVKGMGVSDPDAKKFYAITTLVP...  TGRPEWIWLALGTALMGLGTLYFLVKGMGVSDPDAKKAYAITTLVP...
4   1PY6         A      space     Y43A             Y39A -2.1  TGRPEWIWLALGTALMGLGTLYFLVKGMGVSDPDAKKFYAITTLVP...  TGRPEWIWLALGTALMGLGTLYFLVKGMGVSDPDAKKFAAITTLVP...
'''
for key in config_dict.keys():
    train_embedding = pd.read_pickle(protein_embedding_path + config_dict[key]['train_stab_path'])
    train_origin_df = pd.DataFrame(columns=['seq_before', 'seq_after', 'label','seq_len'])
    train_da_df = pd.DataFrame(columns=['seq_before', 'seq_after', 'label','seq_len'])
    train_da_df2 = pd.DataFrame(columns=['seq_before', 'seq_after', 'label','seq_len'])

    for i in range(train_pkl.shape[0]):
        seq_len = len(train_pkl['seq_before'][i])
        before_head = train_pkl['pdb_id'][i] + '_' + train_pkl['pdb_chain'][i] + '|original'
        after_head = train_pkl['pdb_id'][i] + '_' + train_pkl['pdb_chain'][i] + '|' + train_pkl['shifted_mutation'][i]
        train_origin_df = train_origin_df._append([{'seq_before':train_embedding[before_head], 'seq_after':train_embedding[after_head], 
                                    'label':train_pkl['ddg'][i],'seq_len':seq_len}], ignore_index=True)
        train_da_df = train_da_df._append([{'seq_before':train_embedding[before_head], 'seq_after':train_embedding[after_head], 
                                    'label':train_pkl['ddg'][i],'seq_len':seq_len}], ignore_index=True)
        train_da_df2 = train_da_df2._append([{'seq_before':train_embedding[before_head], 'seq_after':train_embedding[after_head], 
                                    'label':train_pkl['ddg'][i],'seq_len':seq_len}], ignore_index=True)
        train_da_df2 = train_da_df2._append([{'seq_before':train_embedding[before_head], 'seq_after':train_embedding[before_head], 
                                    'label':0,'seq_len':seq_len}], ignore_index=True)

        # reversed items
        train_da_df = train_da_df._append([{'seq_before':train_embedding[after_head], 'seq_after':train_embedding[before_head], 
                                    'label':-train_pkl['ddg'][i],'seq_len':seq_len}], ignore_index=True)
        train_da_df2 = train_da_df2._append([{'seq_before':train_embedding[after_head], 'seq_after':train_embedding[before_head], 
                                    'label':-train_pkl['ddg'][i],'seq_len':seq_len}], ignore_index=True)
        train_da_df2 = train_da_df2._append([{'seq_before':train_embedding[after_head], 'seq_after':train_embedding[after_head], 
                                    'label':0,'seq_len':seq_len}], ignore_index=True)
    
    train_origin_df.to_pickle('../datasets/final/protein_embedding/train_stab_da(ori)_' + key + '.pkl')
    train_da_df.to_pickle('../datasets/final/protein_embedding/train_stab_da(ori+rev)_' + key + '.pkl')
    train_da_df2.to_pickle('../datasets/final/protein_embedding/train_stab_da(ori+rev+non)_' + key + '.pkl')

print(train_da_df.head())
'''
                                          seq_before                                          seq_after  label seq_len
0  [[0.016424907, -0.59978247, 0.37727714, 0.1514...  [[-0.034875847, -0.5766858, 0.11331075, 0.1016...   -0.1     227
1  [[-0.034875847, -0.5766858, 0.11331075, 0.1016...  [[0.016424907, -0.59978247, 0.37727714, 0.1514...    0.1     227
2  [[0.016424907, -0.59978247, 0.37727714, 0.1514...  [[-0.066715546, -0.60104203, 0.54211426, 0.248...   -1.8     227
3  [[-0.066715546, -0.60104203, 0.54211426, 0.248...  [[0.016424907, -0.59978247, 0.37727714, 0.1514...    1.8     227
4  [[0.016424907, -0.59978247, 0.37727714, 0.1514...  [[-0.0068246, -0.5988431, 0.4081417, 0.1343807...   -0.6     227
'''
print(train_da_df2.head())
'''
                                          seq_before                                          seq_after  label seq_len
0  [[0.016424907, -0.59978247, 0.37727714, 0.1514...  [[-0.034875847, -0.5766858, 0.11331075, 0.1016...   -0.1     227
1  [[0.016424907, -0.59978247, 0.37727714, 0.1514...  [[0.016424907, -0.59978247, 0.37727714, 0.1514...    0.0     227
2  [[-0.034875847, -0.5766858, 0.11331075, 0.1016...  [[0.016424907, -0.59978247, 0.37727714, 0.1514...    0.1     227
3  [[-0.034875847, -0.5766858, 0.11331075, 0.1016...  [[-0.034875847, -0.5766858, 0.11331075, 0.1016...    0.0     227
4  [[0.016424907, -0.59978247, 0.37727714, 0.1514...  [[-0.066715546, -0.60104203, 0.54211426, 0.248...   -1.8     227
'''

# generate reversed mcsm test dataset
print(test_reversed_mcsm.head())
'''
  pdb_id pdb_chain uniprot_id mutation shifted_mutation  ddg                                         seq_before                                          seq_after
0   1AFO         A      space     L75A             L14A  1.3           VQLAHHFSEPEITAIIFGVMAGVIGTILLISYGIRRLIKK           VQLAHHFSEPEITLIIFGVMAGVIGTILLISYGIRRLIKK
1   1AFO         A      space     I76A             I15A  1.8           VQLAHHFSEPEITLAIFGVMAGVIGTILLISYGIRRLIKK           VQLAHHFSEPEITLIIFGVMAGVIGTILLISYGIRRLIKK
2   1AFO         A      space     G79A             G18A  1.7           VQLAHHFSEPEITLIIFAVMAGVIGTILLISYGIRRLIKK           VQLAHHFSEPEITLIIFGVMAGVIGTILLISYGIRRLIKK
3   1AFO         A      space     V80A             V19A  0.4           VQLAHHFSEPEITLIIFGAMAGVIGTILLISYGIRRLIKK           VQLAHHFSEPEITLIIFGVMAGVIGTILLISYGIRRLIKK
4   2K73         A      space     A62G             A62G  1.3  MLRFLNQASQGRGAWLLMAFTALALELTALWFQHVMLLKPCVLSIY...  MLRFLNQASQGRGAWLLMAFTALALELTALWFQHVMLLKPCVLSIY...
'''
test_mcsm_reversed_df = pd.DataFrame(columns=['seq_before', 'seq_after', 'label','seq_len'])
for key in config_dict.keys():
    test_embedding = pd.read_pickle(protein_embedding_path + config_dict[key]['test_stab_path'])
    for i in range(test_reversed_mcsm.shape[0]):
        seq_len = len(test_reversed_mcsm['seq_before'][i])
        before_head = test_reversed_mcsm['pdb_id'][i] + '_' + test_reversed_mcsm['pdb_chain'][i] + '|' + test_reversed_mcsm['shifted_mutation'][i]
        after_head = test_reversed_mcsm['pdb_id'][i] + '_' + test_reversed_mcsm['pdb_chain'][i] + '|original'
        test_mcsm_reversed_df = test_mcsm_reversed_df._append([{'seq_before':test_embedding[before_head], 'seq_after':test_embedding[after_head], 
                                    'label':test_reversed_mcsm['ddg'][i],'seq_len':seq_len}], ignore_index=True)
        test_mcsm_reversed_df.to_pickle('../datasets/final/protein_embedding/test_stab_mcsm(rev)_' + key + '.pkl')
print(test_mcsm_reversed_df.head())
'''
                                          seq_before                                          seq_after  label seq_len
0  [[0.35596845, 0.38706288, 0.15608627, 0.216079...  [[0.34166375, 0.38692755, 0.15762639, 0.224924...    1.3      40
1  [[0.336122, 0.3970711, 0.16139467, 0.21426491,...  [[0.34166375, 0.38692755, 0.15762639, 0.224924...    1.8      40
2  [[0.32895786, 0.35713646, 0.14480498, 0.230077...  [[0.34166375, 0.38692755, 0.15762639, 0.224924...    1.7      40
3  [[0.3570439, 0.3902226, 0.15156122, 0.21271954...  [[0.34166375, 0.38692755, 0.15762639, 0.224924...    0.4      40
4  [[0.33413526, 0.41797912, 0.07272053, 0.606040...  [[0.32291368, 0.4135981, 0.07743684, 0.5991041...    1.3     183
'''

# generate reversed test dataset
test_origin_df = pd.DataFrame(columns=['seq_before', 'seq_after', 'label','seq_len'])
test_da_df = pd.DataFrame(columns=['seq_before', 'seq_after', 'label','seq_len'])
test_reversed_df = pd.DataFrame(columns=['seq_before', 'seq_after', 'label','seq_len'])
test_da_df2 = pd.DataFrame(columns=['seq_before', 'seq_after', 'label','seq_len'])

for key in config_dict.keys():
    test_embedding = pd.read_pickle(protein_embedding_path + config_dict[key]['test_stab_path'])
    for i in range(test_pkl.shape[0]):
        before_head = test_pkl['pdb_id'][i] + '_' + test_pkl['pdb_chain'][i] + '|original'
        after_head = test_pkl['pdb_id'][i] + '_' + test_pkl['pdb_chain'][i] + '|' + test_pkl['shifted_mutation'][i]
        test_origin_df = test_origin_df._append([{'seq_before':test_embedding[before_head], 'seq_after':test_embedding[after_head], 
                                    'label':test_pkl['ddg'][i],'seq_len':seq_len}], ignore_index=True)
        test_da_df = test_da_df._append([{'seq_before':test_embedding[before_head], 'seq_after':test_embedding[after_head], 
                                    'label':test_pkl['ddg'][i],'seq_len':seq_len}], ignore_index=True)
        test_da_df2 = test_da_df2._append([{'seq_before':test_embedding[before_head], 'seq_after':test_embedding[before_head], 
                                    'label':0,'seq_len':seq_len}], ignore_index=True)
        # reversed items
        test_da_df = test_da_df._append([{'seq_before':test_embedding[after_head], 'seq_after':test_embedding[before_head], 
                                    'label':-test_pkl['ddg'][i],'seq_len':seq_len}], ignore_index=True)
        test_reversed_df = test_reversed_df._append([{'seq_before':test_embedding[after_head], 'seq_after':test_embedding[before_head], 
                                    'label':-test_pkl['ddg'][i],'seq_len':seq_len}], ignore_index=True)
        test_da_df2 = test_da_df2._append([{'seq_before':test_embedding[after_head], 'seq_after':test_embedding[after_head], 
                                    'label':0,'seq_len':seq_len}], ignore_index=True)
    test_origin_df.to_pickle('../datasets/final/protein_embedding/test_stab_mcsm(ori)_' + key + '.pkl')
    test_da_df.to_pickle('../datasets/final/protein_embedding/test_stab_da(ori+rev)_' + key + '.pkl')
    test_reversed_df.to_pickle('../datasets/final/protein_embedding/test_stab_da(rev)_' + key + '.pkl')
    test_da_df2.to_pickle('../datasets/final/protein_embedding/test_stab_da(non)_' + key + '.pkl')
print(test_da_df.head())
'''
                                          seq_before                                          seq_after  label seq_len
0  [[0.34166375, 0.38692755, 0.15762639, 0.224924...  [[0.35596845, 0.38706288, 0.15608627, 0.216079...   -1.3     183
1  [[0.35596845, 0.38706288, 0.15608627, 0.216079...  [[0.34166375, 0.38692755, 0.15762639, 0.224924...    1.3     183
2  [[0.34166375, 0.38692755, 0.15762639, 0.224924...  [[0.336122, 0.3970711, 0.16139467, 0.21426491,...   -1.8     183
3  [[0.336122, 0.3970711, 0.16139467, 0.21426491,...  [[0.34166375, 0.38692755, 0.15762639, 0.224924...    1.8     183
4  [[0.34166375, 0.38692755, 0.15762639, 0.224924...  [[0.32895786, 0.35713646, 0.14480498, 0.230077...   -1.7     183
'''

# generate mcsm test set
mcsm_test = pd.concat([test_origin_df,test_mcsm_reversed_df])
mcsm_test = mcsm_test.reset_index(drop=True)
mcsm_test.to_pickle('../datasets/final/protein_embedding/test_stab_mcsm(all)_' + key + '.pkl')
print(mcsm_test.head())
'''
                                          seq_before                                          seq_after  label seq_len
0  [[0.34166375, 0.38692755, 0.15762639, 0.224924...  [[0.35596845, 0.38706288, 0.15608627, 0.216079...   -1.3     183
1  [[0.34166375, 0.38692755, 0.15762639, 0.224924...  [[0.336122, 0.3970711, 0.16139467, 0.21426491,...   -1.8     183
2  [[0.34166375, 0.38692755, 0.15762639, 0.224924...  [[0.32895786, 0.35713646, 0.14480498, 0.230077...   -1.7     183
3  [[0.34166375, 0.38692755, 0.15762639, 0.224924...  [[0.3570439, 0.3902226, 0.15156122, 0.21271954...   -0.4     183
4  [[0.32291368, 0.4135981, 0.07743684, 0.5991041...  [[0.33413526, 0.41797912, 0.07272053, 0.606040...   -1.3     183
'''