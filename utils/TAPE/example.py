import torch
import pickle
import numpy as np
import pandas as pd
from tape import ProteinBertModel, TAPETokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pad_or_cut_tensor(tensor, target_length=1024):
    current_length = tensor.size(1)  # 获取当前张量的长度
    if current_length < target_length:
        padding_length = target_length - current_length
        padding = torch.zeros((1, padding_length, tensor.size(2))).to(device)  # 创建填充张量
        padded_tensor = torch.cat((tensor, padding), dim=1)  # 进行填充操作
        return padded_tensor
    elif current_length > target_length:
        cut_tensor = tensor[:, :target_length, :]  # 进行裁剪操作
        return cut_tensor
    else:
        return tensor


model = ProteinBertModel.from_pretrained('bert-base')
model = model.to(device)
tokenizer = TAPETokenizer(vocab='iupac')  # iupac is the vocab for TAPE models, use unirep for the UniRep model
max_length = 8190


def truncate_input(text):
    if len(text) > max_length:
        text = text[:max_length]
    return text


def tape_bert(data):
    with torch.no_grad():
        truncated_data = truncate_input(data)
        token_ids = torch.tensor(np.array([tokenizer.encode(truncated_data)])).to('cuda:0')
        output = model(token_ids)
        sequence_output = output[0]
        sequence_output = sequence_output[:, 1:-1, :]
        sequence_output = torch.squeeze(sequence_output, 0).cpu()
    sequence_output = np.array(sequence_output)
    return sequence_output
