import pandas as pd
import numpy as np

origin_test_file = '../datasets/middlefile/test_stab_df.pkl'
reversed_test_file = '../datasets/cleaned/mCSM_membrane/mcsm_membrane_stability_test_reversed.tsv'

origin_df = pd.read_pickle(origin_test_file)
reversed_df = pd.read_csv(reversed_test_file, sep='\t')

print(origin_df.head())
'''
  pdb_id pdb_chain uniprot_id mutation shifted_mutation  ddg                                         seq_before                                          seq_after
0   1AFO         A      space     L75A             L14A -1.3           VQLAHHFSEPEITLIIFGVMAGVIGTILLISYGIRRLIKK           VQLAHHFSEPEITAIIFGVMAGVIGTILLISYGIRRLIKK
1   1AFO         A      space     I76A             I15A -1.8           VQLAHHFSEPEITLIIFGVMAGVIGTILLISYGIRRLIKK           VQLAHHFSEPEITLAIFGVMAGVIGTILLISYGIRRLIKK
2   1AFO         A      space     G79A             G18A -1.7           VQLAHHFSEPEITLIIFGVMAGVIGTILLISYGIRRLIKK           VQLAHHFSEPEITLIIFAVMAGVIGTILLISYGIRRLIKK
3   1AFO         A      space     V80A             V19A -0.4           VQLAHHFSEPEITLIIFGVMAGVIGTILLISYGIRRLIKK           VQLAHHFSEPEITLIIFGAMAGVIGTILLISYGIRRLIKK
4   2K73         A      space     A62G             A62G -1.3  MLRFLNQASQGRGAWLLMAFTALALELTALWFQHVMLLKPCVLSIY...  MLRFLNQASQGRGAWLLMAFTALALELTALWFQHVMLLKPCVLSIY...
'''
print(reversed_df.head())
'''
   DDG                PDB MUTATION CHAIN
0  1.3  1AFO.mut.L75A.pdb     A75L     A
1  1.8  1AFO.mut.I76A.pdb     A76I     A
2  1.7  1AFO.mut.G79A.pdb     A79G     A
3  0.4  1AFO.mut.V80A.pdb     A80V     A
4  1.3  2K73.mut.A62G.pdb     G62A     A
'''

stab_reversed_df = pd.DataFrame(columns=['pdb_id', 'pdb_chain', 'uniprot_id', 'mutation', 'shifted_mutation', 'ddg', 'seq_before', 'seq_after'])
for i in range(reversed_df.shape[0]):
    pdb_id = reversed_df['PDB'][i].split('.')[0]
    mutation_ori = reversed_df['PDB'][i].split('.')[2]
    shifted_mutation = origin_df.loc[(origin_df['pdb_id']==pdb_id) & (origin_df['mutation']==mutation_ori), 'shifted_mutation'].values[0]
    seq_before = origin_df.loc[(origin_df['pdb_id']==pdb_id) & (origin_df['mutation']==mutation_ori), 'seq_after'].values[0]
    seq_after = origin_df.loc[(origin_df['pdb_id']==pdb_id) & (origin_df['mutation']==mutation_ori), 'seq_before'].values[0]
    #print(shifted_mutation)
    stab_reversed_df = stab_reversed_df._append([{'pdb_id':pdb_id, 'pdb_chain':reversed_df['CHAIN'][i], 'uniprot_id':'space', 
                                    'mutation':mutation_ori, 'shifted_mutation':shifted_mutation, 
                                    'ddg':reversed_df['DDG'][i], 'seq_before':seq_before, 'seq_after':seq_after}], ignore_index=True)
print(stab_reversed_df.head())
'''
  pdb_id pdb_chain uniprot_id mutation shifted_mutation  ddg                                         seq_before                                          seq_after
0   1AFO         A      space     L75A             L14A  1.3           VQLAHHFSEPEITAIIFGVMAGVIGTILLISYGIRRLIKK           VQLAHHFSEPEITLIIFGVMAGVIGTILLISYGIRRLIKK
1   1AFO         A      space     I76A             I15A  1.8           VQLAHHFSEPEITLAIFGVMAGVIGTILLISYGIRRLIKK           VQLAHHFSEPEITLIIFGVMAGVIGTILLISYGIRRLIKK
2   1AFO         A      space     G79A             G18A  1.7           VQLAHHFSEPEITLIIFAVMAGVIGTILLISYGIRRLIKK           VQLAHHFSEPEITLIIFGVMAGVIGTILLISYGIRRLIKK
3   1AFO         A      space     V80A             V19A  0.4           VQLAHHFSEPEITLIIFGAMAGVIGTILLISYGIRRLIKK           VQLAHHFSEPEITLIIFGVMAGVIGTILLISYGIRRLIKK
4   2K73         A      space     A62G             A62G  1.3  MLRFLNQASQGRGAWLLMAFTALALELTALWFQHVMLLKPCVLSIY...  MLRFLNQASQGRGAWLLMAFTALALELTALWFQHVMLLKPCVLSIY...
'''
stab_reversed_df.to_pickle('../datasets/middlefile/test_stab_reversed_df.pkl')