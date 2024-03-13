# generate protein embedding config file
import pandas as pd
import numpy as np
import json
import pickle

output_path = '../datasets/protein_embedding/summary.config'

# ------ generate config file -----
main_dict = {}
main_dict['Esm2_t6'] = {}
main_dict['Esm2_t6']['train_stab_path'] = 'Esm/train_stab_esm2_t6_8M_UR50D_320.pkl'
main_dict['Esm2_t6']['test_stab_path'] = 'Esm/test_stab_esm2_t6_8M_UR50D_320.pkl'
main_dict['Esm2_t6']['params'] = '8M'
main_dict['Esm2_t6']['embedding_dim'] = 320

main_dict['Esm2_t12'] = {}
main_dict['Esm2_t12']['train_stab_path'] = 'Esm/train_stab_esm2_t12_35M_UR50D_480.pkl'
main_dict['Esm2_t12']['test_stab_path'] = 'Esm/test_stab_esm2_t12_35M_UR50D_480.pkl'
main_dict['Esm2_t12']['params'] = '35M'
main_dict['Esm2_t12']['embedding_dim'] = 480

main_dict['Esm2_t30'] = {}
main_dict['Esm2_t30']['train_stab_path'] = 'Esm/train_stab_esm2_t30_150M_UR50D_640.pkl'
main_dict['Esm2_t30']['test_stab_path'] = 'Esm/test_stab_esm2_t30_150M_UR50D_640.pkl'
main_dict['Esm2_t30']['params'] = '150M'
main_dict['Esm2_t30']['embedding_dim'] = 640

main_dict['Esm2_33'] = {}
main_dict['Esm2_33']['train_stab_path'] = 'Esm/train_stab_esm2_t33_650M_UR50D_1280.pkl'
main_dict['Esm2_33']['test_stab_path'] = 'Esm/test_stab_esm2_t33_650M_UR50D_1280.pkl'
main_dict['Esm2_33']['params'] = '650M'
main_dict['Esm2_33']['embedding_dim'] = 1280

main_dict['KeAP'] = {}
main_dict['KeAP']['train_stab_path'] = 'KeAP/train_stab_keap_1024.pkl'
main_dict['KeAP']['test_stab_path'] = 'KeAP/test_stab_keap_1024.pkl'
main_dict['KeAP']['params'] = '420M'
main_dict['KeAP']['embedding_dim'] = 1024

main_dict['TAPE'] = {}
main_dict['TAPE']['train_stab_path'] = 'TAPE/train_stab_bert_768.pkl'
main_dict['TAPE']['test_stab_path'] = 'TAPE/test_stab_bert_768.pkl'
main_dict['TAPE']['params'] = '90M'
main_dict['TAPE']['embedding_dim'] = 768

main_dict['ProtBert'] = {}
main_dict['ProtBert']['train_stab_path'] = 'ProtBert/train_stab_protbert_1024.pkl'
main_dict['ProtBert']['test_stab_path'] = 'ProtBert/test_stab_protbert_1024.pkl'
main_dict['ProtBert']['params'] = '420M'
main_dict['ProtBert']['embedding_dim'] = 1024

main_dict['ProtBert-BFD'] = {}
main_dict['ProtBert-BFD']['train_stab_path'] = 'ProtBert/train_stab_protbert_bfd_1024.pkl'
main_dict['ProtBert-BFD']['test_stab_path'] = 'ProtBert/test_stab_protbert_bfd_1024.pkl'
main_dict['ProtBert-BFD']['params'] = '420M'
main_dict['ProtBert-BFD']['embedding_dim'] = 1024

main_dict['ProtT5-XL-UniRef50'] = {}
main_dict['ProtT5-XL-UniRef50']['train_stab_path'] = 'ProtT5/prot_t5_xl_uniref50/train_stab_prott5_xl_uniref50_1024.pkl'
main_dict['ProtT5-XL-UniRef50']['test_stab_path'] = 'ProtT5/prot_t5_xl_uniref50/test_stab_prott5_xl_uniref50_1024.pkl'
main_dict['ProtT5-XL-UniRef50']['params'] = '1208M'
main_dict['ProtT5-XL-UniRef50']['embedding_dim'] = 1024

main_dict['ProtT5-XL-BFD'] = {}
main_dict['ProtT5-XL-BFD']['train_stab_path'] = 'ProtT5/prot_t5_xl_bfd/train_stab_prott5_xl_bfd_1024.pkl'
main_dict['ProtT5-XL-BFD']['test_stab_path'] = 'ProtT5/prot_t5_xl_bfd/test_stab_prott5_xl_bfd_1024.pkl'
main_dict['ProtT5-XL-BFD']['params'] = '1208M'
main_dict['ProtT5-XL-BFD']['embedding_dim'] = 1024

with open(output_path, 'w+') as w:
    json.dump(main_dict, w, indent=4)

# test loading
test_pkl = pd.read_pickle('../datasets/protein_embedding/Esm/train_stab_esm2_t6_8M_UR50D_320.pkl')
print(test_pkl.keys())
'''
dict_keys(['1PY6_A|original', '1PY6_A|E5A', '1PY6_A|L9A', '1PY6_A|A35P',...])
'''
print(test_pkl['1PY6_A|original'].shape) # (227, 320)

# clean head annotations
for key in main_dict.keys():
    train_embedding = pd.read_pickle('../datasets/protein_embedding/' + main_dict[key]['train_stab_path'])
    test_embedding = pd.read_pickle('../datasets/protein_embedding/' + main_dict[key]['test_stab_path'])
    new_train_embedding = {}
    new_test_embedding = {}
    for sub_key in train_embedding.keys():
        if(len(sub_key.split('|')) == 3):
            new_key = sub_key.split('|')[0] + '|' + sub_key.split('|')[1]
            new_train_embedding[new_key] = train_embedding[sub_key]
        else:
            new_train_embedding[sub_key] = train_embedding[sub_key]
    for sub_key in test_embedding.keys():
        if(len(sub_key.split('|')) == 3):
            new_key = sub_key.split('|')[0] + '|' + sub_key.split('|')[1]
            new_test_embedding[new_key] = test_embedding[sub_key]
        else:
            new_test_embedding[sub_key] = test_embedding[sub_key]
    with open('../datasets/protein_embedding_fix/' + main_dict[key]['train_stab_path'], 'wb') as w:
        pickle.dump(new_train_embedding, w)
    with open('../datasets/protein_embedding_fix/' + main_dict[key]['test_stab_path'], 'wb') as w:
        pickle.dump(new_test_embedding, w)
    #new_train_embedding.to_pickle('../datasets/protein_embedding_fix/' + main_dict[key]['train_stab_path'])
    #new_test_embedding.to_pickle('../datasets/protein_embedding_fix/' + main_dict[key]['test_stab_path'])
# test loading
test_pkl = pd.read_pickle('../datasets/protein_embedding_fix/Esm/train_stab_esm2_t6_8M_UR50D_320.pkl')
print(test_pkl.keys())