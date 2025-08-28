import sys

import torch
from torch.utils.data import Dataset
import numpy as np

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25



def label_smiles(line, smi_ch_ind, MAX_SMI_LEN=100):
    X = np.zeros(MAX_SMI_LEN, dtype=np.int64())
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


def label_sequence(line, smi_ch_ind, MAX_SEQ_LEN=1000):
    X = np.zeros(MAX_SEQ_LEN, np.int64())
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = smi_ch_ind[ch]
    return X


class CustomDataSet(Dataset):
    def __init__(self, pairs, id2index_drug, id2index_protein):
        self.pairs = pairs
        self.id2index_drug = id2index_drug
        self.id2index_protein = id2index_protein

    def __getitem__(self, item):
        sample =  self.pairs[item].strip().split()

        drug_id, protein_id, compoundstr, proteinstr, label = sample[-5], sample[-4], sample[-3], sample[-2], sample[-1]

        if drug_id not in self.id2index_drug and protein_id not in self.id2index_protein:
            print(f"[ERROR] Missing ID: drug_id={drug_id}, protein_id={protein_id}")
            sys.exit(1)  # 退出程序，返回码 1 表示错误退出

        d_index = self.id2index_drug[drug_id]
        p_index = self.id2index_protein[protein_id]

        return d_index, p_index, drug_id, protein_id, compoundstr, proteinstr, label

    def __len__(self):
        return len(self.pairs)


def collate_fn(batch_data):
    N = len(batch_data)
    drug_index, protein_index, drug_idx, protein_idx = [], [], [], []
    compound_max = 100
    protein_max = 1000
    compound_new = torch.zeros((N, compound_max), dtype=torch.long)
    protein_new = torch.zeros((N, protein_max), dtype=torch.long)
    labels_new = torch.zeros(N, dtype=torch.long)
    for i, pair in enumerate(batch_data):
        d_index, p_index, drug_id, protein_id, compoundstr, proteinstr, label = pair[-7], pair[-6], pair[-5], pair[-4], pair[-3], pair[-2], pair[-1]
        drug_index.append(d_index)
        protein_index.append(p_index)

        drug_idx.append(drug_id)
        protein_idx.append(protein_id)

        compoundint = torch.from_numpy(label_smiles(
            compoundstr, CHARISOSMISET, compound_max))
        compound_new[i] = compoundint
        proteinint = torch.from_numpy(label_sequence(
            proteinstr, CHARPROTSET, protein_max))
        protein_new[i] = proteinint
        label = float(label)
        labels_new[i] = int(label)
    return compound_new, protein_new, labels_new, torch.LongTensor(drug_index), torch.LongTensor(protein_index), drug_idx, protein_idx


#
# class CustomDataSet(Dataset):
#     def __init__(self, pairs):
#         self.pairs = pairs
#
#     def __getitem__(self, item):
#         return self.pairs[item]
#
#     def __len__(self):
#         return len(self.pairs)
#
#
# def collate_fn(batch_data):
#     N = len(batch_data)
#     drug_ids, protein_ids = [], []
#     compound_max = 100
#     protein_max = 1000
#     compound_new = torch.zeros((N, compound_max), dtype=torch.long)
#     protein_new = torch.zeros((N, protein_max), dtype=torch.long)
#     labels_new = torch.zeros(N, dtype=torch.long)
#     for i, pair in enumerate(batch_data):
#         pair = pair.strip().split()
#         drug_id, protein_id, compoundstr, proteinstr, label = pair[-5], pair[-4], pair[-3], pair[-2], pair[-1]
#         drug_ids.append(drug_id)
#         protein_ids.append(protein_id)
#         compoundint = torch.from_numpy(label_smiles(
#             compoundstr, CHARISOSMISET, compound_max))
#         compound_new[i] = compoundint
#         proteinint = torch.from_numpy(label_sequence(
#             proteinstr, CHARPROTSET, protein_max))
#         protein_new[i] = proteinint
#         label = float(label)
#         labels_new[i] = int(label)
#     return compound_new, protein_new, labels_new, drug_ids, protein_ids
