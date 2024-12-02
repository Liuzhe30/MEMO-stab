# Load model directly
import gc

import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, EsmModel
from torch.utils.data import Dataset, DataLoader

class ProteinDataset(Dataset):
    def __init__(self, proteins):
        self.proteins = proteins

    def __len__(self):
        return len(self.proteins)

    def __getitem__(self, idx):
        return self.proteins[idx]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("esm2_t33_650M_UR50D")
model = EsmModel.from_pretrained("esm2_t33_650M_UR50D")
model = model.to(device)

tmp_data = pd.read_csv('../bindingdb_protein.csv')
proteins = tmp_data['protein']
protein_dataset = ProteinDataset(proteins)

batch_size = 32
shuffle = False
data_loader = DataLoader(protein_dataset, batch_size=batch_size, shuffle=shuffle)

i = 0
with torch.no_grad():
    for batch in data_loader:
        inputs = tokenizer(batch, padding="max_length", truncation=True, max_length=1024, return_tensors="pt").to(device)
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state.cpu()
        if i > 0:
            stacked_tensor = torch.cat([stacked_tensor, last_hidden_states], 0)
        else:
            stacked_tensor = last_hidden_states
        i = i+1
        print(i)
        del inputs
        del outputs
        del last_hidden_states
        gc.collect()

print(stacked_tensor.shape)
torch.save(stacked_tensor, f'bindingdb_protein_esm2_t33_650M_UR50D.pt')