# -*- coding:utf-8 -*-

import os

import numpy as np
import torch
import math

def get_alpha(epoch, total_decay_epochs, initial_alpha=1.0, final_alpha=0.0, mode='sigmoid'):
    """
    非线性 alpha 衰减计算
    mode: 'linear' | 'cosine' | 'sigmoid'
    """
    progress = min(epoch / total_decay_epochs, 1.0)

    if mode == 'linear':
        alpha = initial_alpha * (1 - progress) + final_alpha * progress
    elif mode == 'cosine':
        alpha = final_alpha + 0.5 * (initial_alpha - final_alpha) * (1 + math.cos(math.pi * progress))
    elif mode == 'sigmoid':
        # 控制sigmoid中心靠近 decay_epochs/2 位置
        k = 10  # 控制曲线陡峭程度
        mid = 0.5  # 中心点
        alpha = final_alpha + (initial_alpha - final_alpha) / (1 + math.exp(k * (progress - mid)))
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    return alpha

def freeze_cnn(model):
    for param in model.Drug_CNNs.parameters():
        param.requires_grad = False
    for param in model.Protein_CNNs.parameters():
        param.requires_grad = False

def freeze_all(model):
    for param in model.parameters():
        param.requires_grad = False


def unfreeze_partial_cnn(model):
    # 解冻 Drug/Protein CNN 的最后一层
    for name, param in model.Drug_CNNs.named_parameters():
        if '2' in name:
            param.requires_grad = True
    for name, param in model.Protein_CNNs.named_parameters():
        if '2' in name:
            param.requires_grad = True


def unfreeze_fusion_path(model):
    for m in [model.fusion_model, model.drug_block, model.pro_block]:
        for p in m.parameters():
            p.requires_grad = True

def group_weight_decay(named_params):
    decay, no_decay = [], []
    for name, param in named_params:
        if param.ndim == 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    return decay, no_decay

def get_optimizer(model, hp):
    # 获取每个模块的参数（只挑 requires_grad 的）
    fusion_named_params = [(n, p) for n, p in model.fusion_model.named_parameters() if p.requires_grad]
    cnn_named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad and 'CNNs' in n]
    drug_named_params = [(n, p) for n, p in model.drug_block.named_parameters() if p.requires_grad]
    pro_named_params = [(n, p) for n, p in model.pro_block.named_parameters() if p.requires_grad]

    decay_fusion, no_decay_fusion = group_weight_decay(fusion_named_params)
    decay_cnn, no_decay_cnn = group_weight_decay(cnn_named_params)
    decay_drug, no_decay_drug = group_weight_decay(drug_named_params)
    decay_pro, no_decay_pro = group_weight_decay(pro_named_params)

    # 构建 optimizer，不同组不同学习率
    optimizer = torch.optim.AdamW([
        {'params': decay_fusion, 'lr': hp.Learning_rate, 'weight_decay': hp.weight_decay},
        {'params': no_decay_fusion, 'lr': hp.Learning_rate, 'weight_decay': 0.0},

        {'params': decay_cnn, 'lr': hp.Learning_rate * 0.05, 'weight_decay': hp.weight_decay},
        {'params': no_decay_cnn, 'lr': hp.Learning_rate * 0.05, 'weight_decay': 0.0},

        {'params': decay_drug, 'lr': hp.Learning_rate, 'weight_decay': hp.weight_decay},
        {'params': no_decay_drug, 'lr': hp.Learning_rate , 'weight_decay': 0.0},

        {'params': decay_pro, 'lr': hp.Learning_rate, 'weight_decay': hp.weight_decay},
        {'params': no_decay_pro, 'lr': hp.Learning_rate, 'weight_decay': 0.0},
    ])

    return optimizer


# def get_optimizer(model, hp):
#     fusion_params = [p for p in model.fusion_model.parameters() if p.requires_grad]
#     partial_cnn_params = [p for name, p in model.named_parameters() if p.requires_grad and 'CNNs' in name]

#     return torch.optim.AdamW([
#         {'params': fusion_params, 'lr': hp.Learning_rate},
#         {'params': partial_cnn_params, 'lr': hp.Learning_rate * 0.05}
#     ])

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, savepath=None, tower_patience=15, fusion_patience=10, verbose=False, delta=0, num_n_fold=0):

        self.tower_patience = tower_patience
        self.fusion_patience = fusion_patience
        self.tower_counter = 0
        self.fusion_counter = 0
        self.verbose = verbose
        self.best_score = -np.inf
        ##########################################
        self.tower_early_stop = False
        self.fusion_early_stop = False
        self.delta = delta
        self.num_n_fold = num_n_fold
        self.savepath = savepath

    def __call__(self, score, model, timestamp, stage="Tower"):
        if self.best_score == -np.inf:
            self.save_checkpoint(score, model, timestamp, stage)
            self.best_score = score

        elif score < self.best_score + self.delta and stage == "Tower":
            self.tower_counter += 1
            print(
                f'EarlyStopping counter of Tower stage: {self.tower_counter} out of {self.tower_patience}')
            if self.tower_counter >= self.tower_patience:
                self.tower_early_stop = True

        elif score < self.best_score + self.delta and stage == "Fusion":
            self.fusion_counter += 1
            print(
                f'EarlyStopping counter of Fusion stage: {self.fusion_counter} out of {self.fusion_patience}')
            if self.fusion_counter >= self.fusion_patience:
                self.fusion_early_stop = True
        else:
            self.save_checkpoint(score, model, timestamp, stage)
            self.best_score = score
            self.tower_counter = 0
            self.fusion_counter= 0

    def save_checkpoint(self, score, model, timestamp, stage):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Have a new best checkpoint: ({self.best_score:.6f} --> {score:.6f}).  Saving model ...')
        save_model = self.savepath + "/" + timestamp
        os.makedirs(save_model, exist_ok=True)
        torch.save(model.state_dict(),  save_model + '/valid_best_checkpoint_'+ stage+'.pth')
