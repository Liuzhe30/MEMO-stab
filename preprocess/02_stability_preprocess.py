import pandas as pd
import numpy as np
import sys 
sys.path.append("..") 
from src import fetchPDBSequence

# step 1: fetch trainset sequence
data_path = '../datasets/cleaned/mCSM_membrane/mcsm_membrane_stability_train.tsv'
PDB_path = '../datasets/raw/mCSM_membrane/pdb_stability/'
df = pd.read_csv(data_path, sep='\t')
print(df.head())
'''
   DDG       PDB MUTATION CHAIN
0 -0.1  1PY6.pdb      E9A     A
1 -1.8  1PY6.pdb     L13A     A
2 -0.6  1PY6.pdb     A39P     A
3 -2.0  1PY6.pdb     F42A     A
4 -2.1  1PY6.pdb     Y43A     A
'''
fetch = fetchPDBSequence.fetchPDBSequence()
stab_df = pd.DataFrame(columns=['pdb_id', 'pdb_chain', 'uniprot_id', 'mutation', 'shifted_mutation', 'ddg', 'seq_before', 'seq_after'])
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
    shifted_mutation = mutation[0] + str(pos+1) + mutation[-1]
    stab_df = stab_df._append([{'pdb_id':pdb_file.split('.')[0], 'pdb_chain':chain, 'uniprot_id':'space', 'mutation':mutation, 'shifted_mutation':shifted_mutation, 
                                    'ddg':df['DDG'][i], 'seq_before':seq_before, 'seq_after':seq_after}], ignore_index=True)

print(stab_df.head())
'''
  pdb_id pdb_chain uniprot_id mutation shifted_mutation  ddg                                         seq_before                                          seq_after
0   1PY6         A      space      E9A              E5A -0.1  TGRPEWIWLALGTALMGLGTLYFLVKGMGVSDPDAKKFYAITTLVP...  TGRPAWIWLALGTALMGLGTLYFLVKGMGVSDPDAKKFYAITTLVP...
1   1PY6         A      space     L13A              L9A -1.8  TGRPEWIWLALGTALMGLGTLYFLVKGMGVSDPDAKKFYAITTLVP...  TGRPEWIWAALGTALMGLGTLYFLVKGMGVSDPDAKKFYAITTLVP...
2   1PY6         A      space     A39P             A35P -0.6  TGRPEWIWLALGTALMGLGTLYFLVKGMGVSDPDAKKFYAITTLVP...  TGRPEWIWLALGTALMGLGTLYFLVKGMGVSDPDPKKFYAITTLVP...
3   1PY6         A      space     F42A             F38A -2.0  TGRPEWIWLALGTALMGLGTLYFLVKGMGVSDPDAKKFYAITTLVP...  TGRPEWIWLALGTALMGLGTLYFLVKGMGVSDPDAKKAYAITTLVP...
4   1PY6         A      space     Y43A             Y39A -2.1  TGRPEWIWLALGTALMGLGTLYFLVKGMGVSDPDAKKFYAITTLVP...  TGRPEWIWLALGTALMGLGTLYFLVKGMGVSDPDAKKFAAITTLVP...
'''
stab_df.to_pickle('../datasets/middlefile/train_stab_df.pkl')

# generate fasta files for run HHblits:
sequence_list = []
fasta_output = '../datasets/annotation/train_stab.fasta'
with open(fasta_output, 'w+') as w:
    for i in range(stab_df.shape[0]):
        if(stab_df['seq_after'][i] not in sequence_list):
            sequence_list.append(stab_df['seq_after'][i])
            w.write('>' + stab_df['pdb_id'][i] + '_' + stab_df['pdb_chain'][i] + '|' + stab_df['shifted_mutation'][i] + '|' + str(stab_df['ddg'][i]) + '\n')
            w.write(stab_df['seq_after'][i] + '\n')

# step 2: split fastas for running HHBlits
fasta_fold = '../datasets/middlefile/fasta/stab/'
train_fasta = '../datasets/annotation/train_stab.fasta'
test_fasta = '../datasets/annotation/test_stab.fasta'
with open(train_fasta, 'r') as r:
    line = r.readline()
    while line:
        pdb_id = line[1:].split('|')[0]
        mutation = line[1:].split('|')[1]
        pos = int(mutation[1:-1]) - 1
        before_aa = mutation[0]
        line = r.readline()
        # fetch mutated sequence
        fastaline = line.strip()
        with open(fasta_fold + pdb_id + '_' + mutation + '.fasta', 'w+') as w:
            w.write('>' + pdb_id + '|' + mutation + '\n')
            w.write(fastaline)
        # generate original sequence
        seq_after = list(fastaline)
        seq_after[pos] = mutation[-1]
        seq_after = ''.join(seq_after)
        with open(fasta_fold + pdb_id + '.fasta', 'w+') as w:
            w.write('>' + pdb_id + '\n')
            w.write(seq_after)
        line = r.readline()
with open(test_fasta, 'r') as r:
    line = r.readline()
    while line:
        pdb_id = line[1:].split('|')[0]
        mutation = line[1:].split('|')[1]
        pos = int(mutation[1:-1]) - 1
        before_aa = mutation[0]
        line = r.readline()
        # fetch mutated sequence
        fastaline = line.strip()
        with open(fasta_fold + pdb_id + '_' + mutation + '.fasta', 'w+') as w:
            w.write('>' + pdb_id + '|' + mutation + '\n')
            w.write(fastaline)
        # generate original sequence
        seq_after = list(fastaline)
        seq_after[pos] = mutation[-1]
        seq_after = ''.join(seq_after)
        with open(fasta_fold + pdb_id + '.fasta', 'w+') as w:
            w.write('>' + pdb_id + '\n')
            w.write(seq_after)
        line = r.readline()