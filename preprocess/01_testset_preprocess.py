import pandas as pd
import numpy as np
import sys 
sys.path.append("..") 
from src import fetchPDBSequence

eyes = np.eye(20)
protein_dict = {'C':eyes[0], 'D':eyes[1], 'S':eyes[2], 'Q':eyes[3], 'K':eyes[4],
    'I':eyes[5], 'P':eyes[6], 'T':eyes[7], 'F':eyes[8], 'N':eyes[9],
    'G':eyes[10], 'H':eyes[11], 'L':eyes[12], 'R':eyes[13], 'W':eyes[14],
    'A':eyes[15], 'V':eyes[16], 'E':eyes[17], 'Y':eyes[18], 'M':eyes[19]}

# testset1: stability prediction
data_path = '../datasets/raw/mCSM_membrane/mcsm_membrane_stability_blind.csv'
PDB_path = '../datasets/raw/mCSM_membrane/pdb_stability/'
df = pd.read_csv(data_path, sep='\t')
print(df.head())
'''
   DDG       PDB MUTATION CHAIN
0 -1.3  1AFO.pdb     L75A     A
1 -1.8  1AFO.pdb     I76A     A
2 -1.7  1AFO.pdb     G79A     A
3 -0.4  1AFO.pdb     V80A     A
4 -1.3  2K73.pdb     A62G     A
'''
fetch = fetchPDBSequence.fetchPDBSequence()
stab_df = pd.DataFrame(columns=['pdb_id', 'pdb_chain', 'uniprot_id', 'mutation', 'ddg', 'seq_before', 'seq_after'])
for i in range(df.shape[0]):
    pdb_file = df['PDB'][i].strip()
    chain = df['CHAIN'][i]
    mutation = df['MUTATION'][i]
    seq_before, poslist = fetch.fetch(PDB_path + pdb_file, chain)
    pos = poslist.index(int(mutation[1:-1]))
    # change AA
    seq_after = list(seq_before)
    seq_after[pos] = mutation[-1]
    seq_after = ''.join(seq_after)
    stab_df = stab_df._append([{'pdb_id':pdb_file.split('.')[0], 'pdb_chain':chain, 'uniprot_id':'space', 'mutation':mutation, 
                                    'ddg':df['DDG'][i], 'seq_before':seq_before, 'seq_after':seq_after}], ignore_index=True)

print(stab_df.head())
'''
  pdb_id pdb_chain uniprot_id mutation  ddg                                         seq_before                                          seq_after
0   1AFO         A      space     L75A -1.3           VQLAHHFSEPEITLIIFGVMAGVIGTILLISYGIRRLIKK           VQLAHHFSEPEITAIIFGVMAGVIGTILLISYGIRRLIKK
1   1AFO         A      space     I76A -1.8           VQLAHHFSEPEITLIIFGVMAGVIGTILLISYGIRRLIKK           VQLAHHFSEPEITLAIFGVMAGVIGTILLISYGIRRLIKK
2   1AFO         A      space     G79A -1.7           VQLAHHFSEPEITLIIFGVMAGVIGTILLISYGIRRLIKK           VQLAHHFSEPEITLIIFAVMAGVIGTILLISYGIRRLIKK
3   1AFO         A      space     V80A -0.4           VQLAHHFSEPEITLIIFGVMAGVIGTILLISYGIRRLIKK           VQLAHHFSEPEITLIIFGAMAGVIGTILLISYGIRRLIKK
4   2K73         A      space     A62G -1.3  MLRFLNQASQGRGAWLLMAFTALALELTALWFQHVMLLKPCVLSIY...  MLRFLNQASQGRGAWLLMAFTALALELTALWFQHVMLLKPCVLSIY...
'''
stab_df.to_pickle('../datasets/middlefile/test_stab_df.pkl')

# generate fasta files:
sequence_list = []
fasta_output = '../datasets/annotation/test_stab.fasta'
with open(fasta_output, 'w+') as w:
    for i in range(stab_df.shape[0]):
        if(stab_df['seq_before'][i] not in sequence_list):
            sequence_list.append(stab_df['seq_before'][i])
            w.write('>' + stab_df['pdb_id'][i] + '_' + stab_df['pdb_chain'][i] + '\n')
            w.write(stab_df['seq_before'][i] + '\n')

# testset2: stability prediction
data_path = '../datasets/raw/mCSM_membrane/mcsm_membrane_pathogenicity_blind.csv'
PDB_path = '../datasets/raw/mCSM_membrane/pdb_pathogenicity/'
df = pd.read_csv(data_path, sep='\t')
print(df.head())
'''
        PDB MUTATION CHAIN       CLASS
0  1P49.pdb    H444R     A  Pathogenic
1  2HYN.pdb      R9C     A  Pathogenic
2  2HYN.pdb      R9H     A  Pathogenic
3  2LNL.pdb    M268L     A      Benign
4  2LZ3.pdb     I22V     A  Pathogenic
'''
fetch = fetchPDBSequence.fetchPDBSequence()
patho_df = pd.DataFrame(columns=['pdb_id', 'pdb_chain', 'uniprot_id', 'mutation', 'class', 'seq_before', 'seq_after'])
for i in range(df.shape[0]):
    pdb_file = df['PDB'][i].strip()
    chain = df['CHAIN'][i]
    mutation = df['MUTATION'][i]
    seq_before, poslist = fetch.fetch(PDB_path + pdb_file, chain)
    pos = poslist.index(int(mutation[1:-1]))
    # change AA
    seq_after = list(seq_before)
    seq_after[pos] = mutation[-1]
    seq_after = ''.join(seq_after)
    patho_df = patho_df._append([{'pdb_id':pdb_file.split('.')[0], 'pdb_chain':chain, 'uniprot_id':'space', 'mutation':mutation, 
                                    'class':df['CLASS'][i], 'seq_before':seq_before, 'seq_after':seq_after}], ignore_index=True)

print(patho_df.head())
'''
  pdb_id pdb_chain uniprot_id mutation       class                                         seq_before                                          seq_after
0   1P49         A      space    H444R  Pathogenic  AASRPNIILVMADDLGIGDPGCYGNKTIRTPNIDRLASGGVKLTQH...  AASRPNIILVMADDLGIGDPGCYGNKTIRTPNIDRLASGGVKLTQH...
1   2HYN         A      space      R9C  Pathogenic  MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICLLLIC...  MEKVQYLTCSAIRRASTIEMPQQARQKLQNLFINFCLILICLLLIC...
2   2HYN         A      space      R9H  Pathogenic  MEKVQYLTRSAIRRASTIEMPQQARQKLQNLFINFCLILICLLLIC...  MEKVQYLTHSAIRRASTIEMPQQARQKLQNLFINFCLILICLLLIC...
3   2LNL         A      space    M268L      Benign  PCMLETETLNKYVVIIAYALVFLLSLLGNSLVMLVILYSRVGRSVT...  PCMLETETLNKYVVIIAYALVFLLSLLGNSLVMLVILYSRVGRSVT...
4   2LZ3         A      space     I22V  Pathogenic                       KGAIIGLMVGGVVIATVIVITLVMLKKK                       KGAIIGLMVGGVVIATVVVITLVMLKKK
'''
patho_df.to_pickle('../datasets/middlefile/test_patho_df.pkl')

# generate fasta files:
sequence_list = []
fasta_output = '../datasets/annotation/test_patho.fasta'
with open(fasta_output, 'w+') as w:
    for i in range(patho_df.shape[0]):
        if(patho_df['seq_before'][i] not in sequence_list):
            sequence_list.append(patho_df['seq_before'][i])
            w.write('>' + patho_df['pdb_id'][i] + '_' + patho_df['pdb_chain'][i] + '\n')
            w.write(patho_df['seq_before'][i] + '\n')