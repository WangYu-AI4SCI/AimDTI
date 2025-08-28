# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.tools import get_alpha

class AimDTI(nn.Module):
    def __init__(self, hp):
        super(AimDTI, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_MAX_LENGTH = hp.drug_max_lengh
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGTH = hp.protein_max_lengh
        self.protein_kernel = hp.protein_kernel
        self.drug_vocab_size = hp.drug_vocab_size
        self.protein_vocab_size = hp.protein_vocab_size
        self.attention_dim = hp.conv * 4
        self.drug_dim_afterCNNs = self.drug_MAX_LENGTH - \
                                  self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3
        self.protein_dim_afterCNNs = self.protein_MAX_LENGTH - \
                                     self.protein_kernel[0] - self.protein_kernel[1] - \
                                     self.protein_kernel[2] + 3
        self.dropout_rate = hp.dropout_rate

        self.drug_embed = nn.Embedding(
            self.drug_vocab_size, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(
            self.protein_vocab_size, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.drug_dim_afterCNNs)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )

        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)


        self.emb_dropout = nn.Dropout(0.1)
        # self.drug_align = nn.Linear(self.conv * 4, self.conv * 4)
        # self.pro_align = nn.Linear(self.conv * 4, self.conv * 4)

        # 结构路径 FC（conv*8 -> 1）
        self.tower_mlp = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.conv * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 1)
        )

        self.drug_block = nn.Sequential(
            nn.Linear(self.conv * 4, self.conv * 4),
            nn.BatchNorm1d(self.conv * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )
        self.pro_block = nn.Sequential(
            nn.Linear(self.conv * 4, self.conv * 4),
            nn.BatchNorm1d(self.conv * 4),
            nn.LeakyReLU(),
            nn.Dropout(0.2)
        )

        self.fusion_model = nn.Sequential(
            nn.Linear(self.conv * 16, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )


    def forward(self, drug, protein, drug_rep=None, pro_rep=None, epoch=None, stage='Tower'):
        # == 1. Embedding
        drug_embed = self.emb_dropout(self.drug_embed(drug)).permute(0, 2, 1)
        protein_embed = self.emb_dropout(self.protein_embed(protein)).permute(0, 2, 1)

        # == 2. CNN
        drugConv = self.Drug_max_pool(self.Drug_CNNs(drug_embed)).squeeze(2)
        proteinConv = self.Protein_max_pool(self.Protein_CNNs(protein_embed)).squeeze(2)

        # == 3. tower-only MLP
        tower_pair = torch.cat([drugConv, proteinConv], dim=1)
        pred_tower = self.tower_mlp(tower_pair)

        # == 4. Fusion
        if epoch is not None and stage == 'Fusion':
            drug_rep = self.drug_block(drug_rep)
            pro_rep = self.pro_block(pro_rep)
            fusion_pair = torch.cat([drugConv, drug_rep, proteinConv, pro_rep], dim=1)
            pred_fusion = self.fusion_model(fusion_pair)
        else:
            pred_fusion = None

        return pred_tower, pred_fusion


class Tower(nn.Module):
    def __init__(self, hp):
        super(Tower, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_MAX_LENGTH = hp.drug_max_lengh
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGTH = hp.protein_max_lengh
        self.protein_kernel = hp.protein_kernel
        self.drug_vocab_size = hp.drug_vocab_size
        self.protein_vocab_size = hp.protein_vocab_size
        self.attention_dim = hp.conv * 4
        self.drug_dim_afterCNNs = self.drug_MAX_LENGTH - \
                                  self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3
        self.protein_dim_afterCNNs = self.protein_MAX_LENGTH - \
                                     self.protein_kernel[0] - self.protein_kernel[1] - \
                                     self.protein_kernel[2] + 3
        self.dropout_rate = hp.dropout_rate

        self.drug_embed = nn.Embedding(
            self.drug_vocab_size, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(
            self.protein_vocab_size, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.drug_dim_afterCNNs)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )

        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)
        self.emb_dropout = nn.Dropout(0.1)


        self.tower_mlp = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.conv * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 1)
        )


    def forward(self, drug, protein, drug_rep=None, pro_rep=None, epoch=None, stage='Tower'):
        # == 1. Embedding
        drug_embed = self.emb_dropout(self.drug_embed(drug)).permute(0, 2, 1)
        protein_embed = self.emb_dropout(self.protein_embed(protein)).permute(0, 2, 1)

        # == 2. CNN
        drugConv = self.Drug_max_pool(self.Drug_CNNs(drug_embed)).squeeze(2)
        proteinConv = self.Protein_max_pool(self.Protein_CNNs(protein_embed)).squeeze(2)

        # == 3. tower-only MLP
        tower_pair = torch.cat([drugConv, proteinConv], dim=1)
        pred_tower = self.tower_mlp(tower_pair)

        return pred_tower, 0

class GEDTI(nn.Module):
    def __init__(self, hp):
        super(GEDTI, self).__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_MAX_LENGTH = hp.drug_max_lengh
        self.drug_kernel = hp.drug_kernel
        self.protein_MAX_LENGTH = hp.protein_max_lengh
        self.protein_kernel = hp.protein_kernel
        self.drug_vocab_size = hp.drug_vocab_size
        self.protein_vocab_size = hp.protein_vocab_size
        self.attention_dim = hp.conv * 4
        self.drug_dim_afterCNNs = self.drug_MAX_LENGTH - \
                                  self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3
        self.protein_dim_afterCNNs = self.protein_MAX_LENGTH - \
                                     self.protein_kernel[0] - self.protein_kernel[1] - \
                                     self.protein_kernel[2] + 3
        self.dropout_rate = hp.dropout_rate

        self.drug_embed = nn.Embedding(
            self.drug_vocab_size, self.dim, padding_idx=0)
        self.protein_embed = nn.Embedding(
            self.protein_vocab_size, self.dim, padding_idx=0)

        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )
        self.Drug_max_pool = nn.MaxPool1d(self.drug_dim_afterCNNs)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )

        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)


        self.emb_dropout = nn.Dropout(0.1)

        # 结构路径 FC（conv*8 -> 1）
        self.tower_mlp = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.conv * 8, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(0.4),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 1)
        )

        self.fusion_model = nn.Sequential(
            nn.Linear(self.conv * 16, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )


    def forward(self, drug, protein, drug_rep=None, pro_rep=None, epoch=None, stage='Tower'):
        # == 1. Embedding
        drug_embed = self.emb_dropout(self.drug_embed(drug)).permute(0, 2, 1)
        protein_embed = self.emb_dropout(self.protein_embed(protein)).permute(0, 2, 1)

        # == 2. CNN
        drugConv = self.Drug_max_pool(self.Drug_CNNs(drug_embed)).squeeze(2)
        proteinConv = self.Protein_max_pool(self.Protein_CNNs(protein_embed)).squeeze(2)

        # == 3. tower-only MLP
        tower_pair = torch.cat([drugConv, proteinConv], dim=1)
        pred_tower = self.tower_mlp(tower_pair)

        # == 4. Fusion
        if epoch is not None and stage == 'Fusion':
            fusion_pair = torch.cat([drugConv, drug_rep, proteinConv, pro_rep], dim=1)
            pred_fusion = self.fusion_model(fusion_pair)
        else:
            pred_fusion = None

        return pred_tower, pred_fusion


