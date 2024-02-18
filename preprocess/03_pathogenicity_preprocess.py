import pandas as pd
import numpy as np
import sys 

# step 1: generate training set
fasta_file = '../datasets/annotation/train_patho.fasta'
patho_df = pd.DataFrame(columns=['pdb_id', 'pdb_chain', 'uniprot_id', 'mutation', 'shifted_mutation', 'seq_before', 'seq_after', 'class'])

with open(fasta_file,'r') as r:
    line = r.readline()
    while line:
        # fasta head
        pdb_id = 'space'
        pdb_chain = 'space'
        uniprot_id = line[1:].split('|')[0]
        mutation = line[1:].split('|')[1]
        label = line[1:].split('|')[2]
        shifted_mutation = mutation # no shifts
        pos = int(mutation[1:-1]) - 1
        # read sequence
        line = r.readline()
        seq_before = line.strip()
        # change AA
        seq_after = list(seq_before)
        seq_after[pos] = mutation[-1]
        seq_after = ''.join(seq_after)
        patho_df = patho_df._append([{'pdb_id':pdb_id, 'pdb_chain':pdb_chain, 'uniprot_id':uniprot_id, 'mutation':mutation, 'shifted_mutation':shifted_mutation, 
                                     'seq_before':seq_before, 'seq_after':seq_after, 'class':label}], ignore_index=True)
        line = r.readline()

print(patho_df.head())
'''

'''
patho_df.to_pickle('../datasets/middlefile/train_patho_df.pkl')

# generate processed fasta file
sequence_list = []
fasta_output = '../datasets/processed/train_patho.fasta'
with open(fasta_output, 'w+') as w:
    for i in range(patho_df.shape[0]):
        if(patho_df['seq_before'][i] not in sequence_list):
            sequence_list.append(patho_df['seq_before'][i])
            w.write('>' + patho_df['pdb_id'][i] + '_' + patho_df['pdb_chain'][i] + '|original|\n')
            w.write(patho_df['seq_before'][i] + '\n')
        if(patho_df['seq_after'][i] not in sequence_list):
            sequence_list.append(patho_df['seq_after'][i])
            w.write('>' + patho_df['pdb_id'][i] + '_' + patho_df['pdb_chain'][i] + '|' + patho_df['shifted_mutation'][i] + '|' + str(patho_df['ddg'][i]) + '\n')
            w.write(patho_df['seq_after'][i] + '\n')


# step 2: split fastas for running HHBlits
fasta_fold = '../datasets/middlefile/fasta/patho/'
train_fasta = '../datasets/processed/train_patho.fasta'
test_fasta = '../datasets/processed/test_patho.fasta'
with open(train_fasta, 'r') as r:
    line = r.readline()
    while line:
        # process original sequences
        if(line[1:].split('|')[1] == 'original'):
            pdb_id = line[1:].split('|')[0]
            line = r.readline()
            # fetch mutated sequence
            fastaline = line.strip()
            with open(fasta_fold + pdb_id + '.fasta', 'w+') as w:
                w.write('>' + pdb_id + '\n')
                w.write(fastaline)
            line = r.readline()
        else:
            pdb_id = line[1:].split('|')[0]
            mutation = line[1:].split('|')[1]
            line = r.readline()
            # fetch mutated sequence
            fastaline = line.strip()
            with open(fasta_fold + pdb_id + '_' + mutation + '.fasta', 'w+') as w:
                w.write('>' + pdb_id + '|' + mutation + '\n')
                w.write(fastaline)
            line = r.readline()
with open(test_fasta, 'r') as r:
    line = r.readline()
    while line:
        # process original sequences
        if(line[1:].split('|')[1] == 'original'):
            pdb_id = line[1:].split('|')[0]
            line = r.readline()
            # fetch mutated sequence
            fastaline = line.strip()
            with open(fasta_fold + pdb_id + '.fasta', 'w+') as w:
                w.write('>' + pdb_id + '\n')
                w.write(fastaline)
            line = r.readline()
        else:
            pdb_id = line[1:].split('|')[0]
            mutation = line[1:].split('|')[1]
            line = r.readline()
            # fetch mutated sequence
            fastaline = line.strip()
            with open(fasta_fold + pdb_id + '_' + mutation + '.fasta', 'w+') as w:
                w.write('>' + pdb_id + '|' + mutation + '\n')
                w.write(fastaline)
            line = r.readline()