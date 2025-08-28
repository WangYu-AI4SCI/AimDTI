# -*- coding:utf-8 -*-
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, f1_score)

from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from config import hyperparameter
from model import AimDTI, Tower, GEDTI
from utils.DataPrepare import get_kfold_data, shuffle_dataset
from utils.DataSetsFunction import CustomDataSet, collate_fn
from utils.tools import EarlyStopping, freeze_cnn, freeze_all, unfreeze_partial_cnn, unfreeze_fusion_path, get_optimizer
from LossFunction import CELoss, PolyLoss, FocalLoss
from utils.TestModel import test_model
from utils.ShowResult import show_result
import datetime
from torch.utils.tensorboard import SummaryWriter
import os

def train(SEED, DATASET, MODEL,  LOSS):
    '''set random seed'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    # cudnn.deterministic = True

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(DEVICE)
    '''init hyperparameters'''
    hp = hyperparameter(DATASET, MODEL)

    '''load dataset from text file'''
    assert DATASET in ["DrugBank", "KIBA", "Davis", "Enzyme", "GPCRs", "ion_channel"]
    print("Train in " + DATASET)
    print("load data")
    dir_train = hp.dir_train
    with open(dir_train, "r") as f:
        train_data_list = f.read().strip().split('\n')
    print("train data load finished")
    print('Number of Train set: {}'.format(len(train_data_list)))
    '''shuffle data'''
    print("data shuffle")
    train_data_list = shuffle_dataset(train_data_list, SEED)

    dir_valid = hp.dir_valid
    with open(dir_valid, "r") as f:
        valid_data_list = f.read().strip().split('\n')
    print("valid data load finished")
    print('Number of Valid set: {}'.format(len(valid_data_list)))

    dir_test = hp.dir_test
    with open(dir_test, "r") as f:
        test_data_list = f.read().strip().split('\n')
    print("test data load finished")
    print('Number of Test set: {}'.format(len(test_data_list)))


    '''set loss function weight'''
    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(DEVICE)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(DEVICE)
    else:
        weight_loss = None

    # 加载 embeddingDataPrepare/embedding/tmp/index2id_drug_dict.json
    drug_rep = np.load(hp.embedding_path + '/best_user_embeddings.npy')  # [num_drugs, dim]
    pro_rep = np.load(hp.embedding_path + '/best_item_embeddings.npy')  # [num_proteins, dim]
    if isinstance(drug_rep, np.ndarray):
        drug_rep = torch.from_numpy(drug_rep).float().to(DEVICE)
    if isinstance(pro_rep, np.ndarray):
        pro_rep = torch.from_numpy(pro_rep).float().to(DEVICE)

    with open(hp.embedding_path + '/index2id_user_dict.json') as f:
        index2id_drug = json.load(f)
    id2index_drug = {v: int(k) for k, v in index2id_drug.items()}
    with open(hp.embedding_path + '/index2id_item_dict.json') as f:
        index2id_protein = json.load(f)
    id2index_protein = {v: int(k) for k, v in index2id_protein.items()}


    '''metrics'''
    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []

    # train_dataset = CustomDataSet(train_data_list)
    # valid_dataset = CustomDataSet(valid_data_list)
    # test_dataset = CustomDataSet(test_data_list)
    train_dataset = CustomDataSet(train_data_list, id2index_drug, id2index_protein)
    valid_dataset = CustomDataSet(valid_data_list, id2index_drug, id2index_protein)
    test_dataset = CustomDataSet(test_data_list, id2index_drug, id2index_protein)
    train_size = len(train_dataset)

    train_dataset_loader = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
                                      collate_fn=collate_fn, drop_last=True)
    valid_dataset_loader = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                      collate_fn=collate_fn, drop_last=True)
    test_dataset_loader = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                     collate_fn=collate_fn, drop_last=True)

    model = AimDTI(hp).to(DEVICE)
    print(MODEL)
    """Initialize weights"""
    weight_p, bias_p = [], []
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]

    """create optimizer and scheduler"""
    optimizer = optim.AdamW(
        [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)

    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False,
                                            step_size_up=train_size // hp.Batch_size)


    if LOSS == 'PolyLoss':
        Loss = PolyLoss(weight_loss=weight_loss,
                        DEVICE=DEVICE, epsilon=hp.loss_epsilon)
    elif LOSS == 'FocalLoss':
        Loss = FocalLoss(alpha=hp.alpha, gamma=hp.gamma)
    else:
        Loss = CELoss(weight_CE=weight_loss, DEVICE=DEVICE)

    """Output files"""
    save_path = hp.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    early_stopping = EarlyStopping(
        savepath=save_path, tower_patience=hp.Tower_patience, fusion_patience=hp.Fusion_patience, verbose=True, delta=0)
    stage = "Tower"

    writer = SummaryWriter(log_dir=os.path.join('runs', f'{MODEL}_{DATASET}_{LOSS}_{timestamp}'))
    """Start training."""
    print('Training...')
    tower_stage_ended = False
    for epoch in range(1, hp.Epoch + 1):
        if (early_stopping.tower_early_stop or epoch == hp.Tower_ending) and not tower_stage_ended:
            model.load_state_dict(torch.load(early_stopping.savepath + "/" + timestamp + '/valid_best_checkpoint_Tower.pth'))
            freeze_all(model)
            unfreeze_partial_cnn(model)
            unfreeze_fusion_path(model)
            optimizer = get_optimizer(model, hp)
            # scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate * 10,
            #                                         cycle_momentum=False,
            #                                         step_size_up=train_size // hp.Batch_size)
            # CosineAnnealingLR（余弦退火学习率调度器）
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=hp.Epoch * (train_size // hp.Batch_size),
                eta_min=1e-5
            )
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('start fusion stage...')
            stage = "Fusion"
            tower_end_epoch = epoch
            early_stopping.tower_early_stop = False
            tower_stage_ended = True
            early_stopping.best_score = -np.inf

        if early_stopping.fusion_early_stop:
            fusion_end_epoch = epoch
            break

        train_pbar = tqdm(
            enumerate(BackgroundGenerator(train_dataset_loader)),
            total=len(train_dataset_loader))

        """train"""
        train_losses_in_epoch = []
        model.train()
        for train_i, train_data in train_pbar:
            train_compounds, train_proteins, train_labels, drug_index, protein_index, _, _ = train_data
            train_compounds = train_compounds.to(DEVICE)
            train_proteins = train_proteins.to(DEVICE)
            train_labels = train_labels.to(DEVICE)
            batch_drug_rep = drug_rep[drug_index].to(DEVICE)
            batch_pro_rep = pro_rep[protein_index].to(DEVICE)

            optimizer.zero_grad()
            # predicted_interaction = model(train_compounds, train_proteins).squeeze(1)
            # predicted_interaction= model(train_compounds, train_proteins, batch_drug_rep, batch_pro_rep, epoch=epoch)

            pred_tower, pred_fusion = model(train_compounds, train_proteins, batch_drug_rep, batch_pro_rep, epoch, stage)

            if stage == "Tower":
                loss = Loss(pred_tower.squeeze(1), train_labels.float())
            else:
                loss = Loss(pred_fusion.squeeze(1), train_labels.float())


            train_losses_in_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss_a_epoch = np.average(train_losses_in_epoch)  # 一次epoch的平均训练loss

        """valid"""
        valid_pbar = tqdm(
            enumerate(BackgroundGenerator(valid_dataset_loader)),
            total=len(valid_dataset_loader))
        valid_losses_in_epoch = []
        model.eval()
        Y, P, S = [], [], []
        with torch.no_grad():
            for valid_i, valid_data in valid_pbar:

                valid_compounds, valid_proteins, valid_labels, drug_index, protein_index, _, _ = valid_data

                valid_compounds = valid_compounds.to(DEVICE)
                valid_proteins = valid_proteins.to(DEVICE)
                valid_labels = valid_labels.to(DEVICE)
                batch_drug_rep = drug_rep[drug_index].to(DEVICE)
                batch_pro_rep = pro_rep[protein_index].to(DEVICE)


                pred_tower, pred_fusion = model(valid_compounds, valid_proteins, batch_drug_rep, batch_pro_rep, epoch, stage)

                if stage == "Tower":
                    valid_loss = Loss(pred_tower.squeeze(1), valid_labels.float())
                    valid_scores = torch.sigmoid(pred_tower).detach().cpu().numpy()
                else:
                    valid_loss = Loss(pred_fusion.squeeze(1), valid_labels.float())
                    valid_scores = torch.sigmoid(pred_fusion).detach().cpu().numpy()

                valid_predictions = (valid_scores >= 0.5).astype(int)
                valid_labels = valid_labels.to('cpu').data.numpy()
                valid_losses_in_epoch.append(valid_loss.item())
                Y.extend(valid_labels)
                P.extend(valid_predictions)
                S.extend(valid_scores)

        Precision_dev = precision_score(Y, P)
        Reacll_dev = recall_score(Y, P)
        Accuracy_dev = accuracy_score(Y, P)
        F1_dev = f1_score(Y, P)
        AUC_dev = roc_auc_score(Y, S)
        tpr, fpr, _ = precision_recall_curve(Y, S)
        PRC_dev = auc(fpr, tpr)
        valid_loss_a_epoch = np.average(valid_losses_in_epoch)

        epoch_len = len(str(hp.Epoch))
        print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                     f'train_loss: {train_loss_a_epoch:.5f} ' +
                     f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                     f'valid_AUC: {AUC_dev:.5f} ' +
                     f'valid_PRC: {PRC_dev:.5f} ' +
                     f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                     f'valid_Precision: {Precision_dev:.5f} ' +
                     f'valid_Reacll: {Reacll_dev:.5f} '+
                     f'valid_F1_score: {F1_dev:.5f} ' )
        print(print_msg)
        writer.add_scalar('Loss/train', train_loss_a_epoch, epoch)
        writer.add_scalar('Loss/valid', valid_loss_a_epoch, epoch)
        writer.add_scalar('Metric/AUC', AUC_dev, epoch)
        writer.add_scalar('Metric/PRC', PRC_dev, epoch)
        writer.add_scalar('Metric/Accuracy', Accuracy_dev, epoch)
        writer.add_scalar('Metric/Precision', Precision_dev, epoch)
        writer.add_scalar('Metric/Recall', Reacll_dev, epoch)
        writer.add_scalar('Metric/F1_score', F1_dev, epoch)
        writer.add_scalar('Metric/LR', optimizer.param_groups[0]['lr'], epoch)

        '''save checkpoint and make decision when early stop'''

        # early_stopping(valid_loss_a_epoch, model, timestamp, stage)
        early_stopping(F1_dev, model, timestamp, stage)

    ##########################################
    '''load best checkpoint'''
    model.load_state_dict(torch.load(
        early_stopping.savepath + "/" + timestamp + '/valid_best_checkpoint_Tower.pth'))

    '''test model'''
    trainset_test_stable_results_s, _, _, _, _, _, _ = test_model(
        model, drug_rep, pro_rep, train_dataset_loader, save_path, timestamp, DATASET, Loss, DEVICE,  dataset_class="Train", stage = "Tower", FOLD_NUM=1)
    validset_test_stable_results_s, _, _, _, _, _, _ = test_model(
        model, drug_rep, pro_rep, valid_dataset_loader, save_path, timestamp, DATASET, Loss, DEVICE, dataset_class="Valid", stage = "Tower", FOLD_NUM=1)
    testset_test_stable_results_s, Accuracy_test, Precision_test, Recall_test, F1_test, AUC_test, PRC_test = test_model(
        model, drug_rep, pro_rep, test_dataset_loader, save_path, timestamp, DATASET, Loss, DEVICE, dataset_class="Test", stage = "Tower", FOLD_NUM=1)

    model.load_state_dict(torch.load(
        early_stopping.savepath + "/" + timestamp + '/valid_best_checkpoint_Fusion.pth'))
    trainset_test_stable_results_f, _, _, _, _, _, _ = test_model(
        model, drug_rep, pro_rep, train_dataset_loader, save_path, timestamp, DATASET, Loss, DEVICE,  dataset_class="Train", stage = "Fusion", FOLD_NUM=1)
    validset_test_stable_results_f, _, _, _, _, _, _ = test_model(
        model, drug_rep, pro_rep, valid_dataset_loader, save_path, timestamp, DATASET, Loss, DEVICE, dataset_class="Valid", stage = "Fusion", FOLD_NUM=1)
    testset_test_stable_results_f, Accuracy_test, Precision_test, Recall_test, F1_test, AUC_test, PRC_test = test_model(
        model, drug_rep, pro_rep, test_dataset_loader, save_path, timestamp, DATASET, Loss, DEVICE, dataset_class="Test", stage = "Fusion", FOLD_NUM=1)

    with open(save_path + '/' + "The_results_of_whole_dataset.txt", 'a') as f:
        f.write("Final results of ")
        f.write(f"{timestamp}\n")
        f.write(f"{MODEL}\t{DATASET}\t{LOSS}\n")
        for k, v in vars(hp).items():
            if k == 'Batch_size' or k== 'alpha' or k== 'gamma':
                f.write(f"{k}: {v}\n")
        f.write("Tower Model: \n")
        f.write(f"Ending Epoch at {tower_end_epoch}\n")
        f.write(trainset_test_stable_results_s + '\n')
        f.write(validset_test_stable_results_s + '\n')
        f.write(testset_test_stable_results_s + '\n')
        f.write("Fusion Model: \n")
        f.write(f"Final Ending Epoch at {fusion_end_epoch}\n")
        f.write(trainset_test_stable_results_f + '\n')
        f.write(validset_test_stable_results_f + '\n')
        f.write(testset_test_stable_results_f + '\n\n')

    writer.close()



def tower_train(SEED, DATASET, MODEL,  LOSS):
    '''set random seed'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    # cudnn.deterministic = True

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(DEVICE)
    '''init hyperparameters'''
    hp = hyperparameter(DATASET)

    '''load dataset from text file'''
    assert DATASET in ["DrugBank", "KIBA", "Davis", "Enzyme", "GPCRs", "ion_channel"]
    print("Train in " + DATASET)
    print("load data")
    dir_train = hp.dir_train
    with open(dir_train, "r") as f:
        train_data_list = f.read().strip().split('\n')
    print("train data load finished")
    print('Number of Train set: {}'.format(len(train_data_list)))
    '''shuffle data'''
    print("data shuffle")
    train_data_list = shuffle_dataset(train_data_list, SEED)

    dir_valid = hp.dir_valid
    with open(dir_valid, "r") as f:
        valid_data_list = f.read().strip().split('\n')
    print("valid data load finished")
    print('Number of Valid set: {}'.format(len(valid_data_list)))

    dir_test = hp.dir_test
    with open(dir_test, "r") as f:
        test_data_list = f.read().strip().split('\n')
    print("test data load finished")
    print('Number of Test set: {}'.format(len(test_data_list)))


    '''set loss function weight'''
    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(DEVICE)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(DEVICE)
    else:
        weight_loss = None

    # 加载 embeddingDataPrepare/embedding/tmp/index2id_drug_dict.json
    drug_rep = np.load(hp.embedding_path + '/best_user_embeddings.npy')  # [num_drugs, dim]
    pro_rep = np.load(hp.embedding_path + '/best_item_embeddings.npy')  # [num_proteins, dim]
    if isinstance(drug_rep, np.ndarray):
        drug_rep = torch.from_numpy(drug_rep).float().to(DEVICE)
    if isinstance(pro_rep, np.ndarray):
        pro_rep = torch.from_numpy(pro_rep).float().to(DEVICE)

    with open(hp.embedding_path + '/index2id_user_dict.json') as f:
        index2id_drug = json.load(f)
    id2index_drug = {v: int(k) for k, v in index2id_drug.items()}
    with open(hp.embedding_path + '/index2id_item_dict.json') as f:
        index2id_protein = json.load(f)
    id2index_protein = {v: int(k) for k, v in index2id_protein.items()}


    '''metrics'''
    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []

    # train_dataset = CustomDataSet(train_data_list)
    # valid_dataset = CustomDataSet(valid_data_list)
    # test_dataset = CustomDataSet(test_data_list)
    train_dataset = CustomDataSet(train_data_list, id2index_drug, id2index_protein)
    valid_dataset = CustomDataSet(valid_data_list, id2index_drug, id2index_protein)
    test_dataset = CustomDataSet(test_data_list, id2index_drug, id2index_protein)
    train_size = len(train_dataset)

    train_dataset_loader = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
                                      collate_fn=collate_fn, drop_last=True)
    valid_dataset_loader = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                      collate_fn=collate_fn, drop_last=True)
    test_dataset_loader = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                     collate_fn=collate_fn, drop_last=True)

    model = Tower(hp).to(DEVICE)
    print(MODEL)
    """Initialize weights"""
    weight_p, bias_p = [], []
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]

    """create optimizer and scheduler"""
    optimizer = optim.AdamW(
        [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)

    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False,
                                            step_size_up=train_size // hp.Batch_size)

    if LOSS == 'PolyLoss':
        Loss = PolyLoss(weight_loss=weight_loss,
                        DEVICE=DEVICE, epsilon=hp.loss_epsilon)
    elif LOSS == 'FocalLoss':
        Loss = FocalLoss(alpha=hp.alpha, gamma=hp.gamma)
    else:
        Loss = CELoss(weight_CE=weight_loss, DEVICE=DEVICE)

    """Output files"""
    save_path = hp.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    early_stopping = EarlyStopping(
        savepath=save_path, tower_patience=hp.Tower_patience, fusion_patience=hp.Fusion_patience, verbose=True, delta=0)
    stage = "Tower"

    writer = SummaryWriter(log_dir=os.path.join('runs', f'{MODEL}_{DATASET}_{LOSS}_{timestamp}'))
    """Start training."""
    print('Training...')
    tower_end_epoch = hp.Epoch
    for epoch in range(1, hp.Epoch + 1):
        if early_stopping.tower_early_stop:
            tower_end_epoch = epoch
            break
        train_pbar = tqdm(
            enumerate(BackgroundGenerator(train_dataset_loader)),
            total=len(train_dataset_loader))

        """train"""
        train_losses_in_epoch = []
        model.train()

        for train_i, train_data in train_pbar:
            train_compounds, train_proteins, train_labels, drug_index, protein_index, _, _ = train_data
            train_compounds = train_compounds.to(DEVICE)
            train_proteins = train_proteins.to(DEVICE)
            train_labels = train_labels.to(DEVICE)
            batch_drug_rep = drug_rep[drug_index].to(DEVICE)
            batch_pro_rep = pro_rep[protein_index].to(DEVICE)

            optimizer.zero_grad()
            # predicted_interaction = model(train_compounds, train_proteins).squeeze(1)
            # predicted_interaction= model(train_compounds, train_proteins, batch_drug_rep, batch_pro_rep, epoch=epoch)

            pred_tower, _ = model(train_compounds, train_proteins, batch_drug_rep, batch_pro_rep, epoch, stage)

            loss = Loss(pred_tower.squeeze(1), train_labels.float())

            train_losses_in_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss_a_epoch = np.average(train_losses_in_epoch)  # 一次epoch的平均训练loss

        """valid"""
        valid_pbar = tqdm(
            enumerate(BackgroundGenerator(valid_dataset_loader)),
            total=len(valid_dataset_loader))
        valid_losses_in_epoch = []
        model.eval()
        Y, P, S = [], [], []
        with torch.no_grad():
            for valid_i, valid_data in valid_pbar:

                valid_compounds, valid_proteins, valid_labels, drug_index, protein_index, _, _ = valid_data

                valid_compounds = valid_compounds.to(DEVICE)
                valid_proteins = valid_proteins.to(DEVICE)
                valid_labels = valid_labels.to(DEVICE)
                batch_drug_rep = drug_rep[drug_index].to(DEVICE)
                batch_pro_rep = pro_rep[protein_index].to(DEVICE)

                pred_tower, _ = model(valid_compounds, valid_proteins, batch_drug_rep, batch_pro_rep, epoch, stage)

                valid_loss = Loss(pred_tower.squeeze(1), valid_labels.float())
                valid_scores = torch.sigmoid(pred_tower).detach().cpu().numpy()

                valid_predictions = (valid_scores >= 0.5).astype(int)
                valid_labels = valid_labels.to('cpu').data.numpy()
                valid_losses_in_epoch.append(valid_loss.item())
                Y.extend(valid_labels)
                P.extend(valid_predictions)
                S.extend(valid_scores)

        Precision_dev = precision_score(Y, P)
        Reacll_dev = recall_score(Y, P)
        Accuracy_dev = accuracy_score(Y, P)
        F1_dev = f1_score(Y, P)
        AUC_dev = roc_auc_score(Y, S)
        tpr, fpr, _ = precision_recall_curve(Y, S)
        PRC_dev = auc(fpr, tpr)
        valid_loss_a_epoch = np.average(valid_losses_in_epoch)

        epoch_len = len(str(hp.Epoch))
        print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                     f'train_loss: {train_loss_a_epoch:.5f} ' +
                     f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                     f'valid_AUC: {AUC_dev:.5f} ' +
                     f'valid_PRC: {PRC_dev:.5f} ' +
                     f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                     f'valid_Precision: {Precision_dev:.5f} ' +
                     f'valid_Reacll: {Reacll_dev:.5f} '+
                     f'valid_F1_score: {F1_dev:.5f} ' )
        print(print_msg)
        writer.add_scalar('Loss/train', train_loss_a_epoch, epoch)
        writer.add_scalar('Loss/valid', valid_loss_a_epoch, epoch)
        writer.add_scalar('Metric/AUC', AUC_dev, epoch)
        writer.add_scalar('Metric/PRC', PRC_dev, epoch)
        writer.add_scalar('Metric/Accuracy', Accuracy_dev, epoch)
        writer.add_scalar('Metric/Precision', Precision_dev, epoch)
        writer.add_scalar('Metric/Recall', Reacll_dev, epoch)
        writer.add_scalar('Metric/F1_score', F1_dev, epoch)
        writer.add_scalar('Metric/LR', optimizer.param_groups[0]['lr'], epoch)

        '''save checkpoint and make decision when early stop'''

        # early_stopping(valid_loss_a_epoch, model, timestamp, stage)
        early_stopping(F1_dev, model, timestamp, stage)

    ##########################################
    '''load best checkpoint'''
    model.load_state_dict(torch.load(
        early_stopping.savepath + "/" + timestamp + '/valid_best_checkpoint_Tower.pth'))

    '''test model'''
    trainset_test_stable_results_s, _, _, _, _, _, _ = test_model(
        model, drug_rep, pro_rep, train_dataset_loader, save_path, timestamp, DATASET, Loss, DEVICE,  dataset_class="Train", stage = "Tower", FOLD_NUM=1)
    validset_test_stable_results_s, _, _, _, _, _, _ = test_model(
        model, drug_rep, pro_rep, valid_dataset_loader, save_path, timestamp, DATASET, Loss, DEVICE, dataset_class="Valid", stage = "Tower", FOLD_NUM=1)
    testset_test_stable_results_s, Accuracy_test, Precision_test, Recall_test, F1_test, AUC_test, PRC_test = test_model(
        model, drug_rep, pro_rep, test_dataset_loader, save_path, timestamp, DATASET, Loss, DEVICE, dataset_class="Test", stage = "Tower", FOLD_NUM=1)

    with open(save_path + '/' + "The_tower_results_of_whole_dataset.txt", 'a') as f:
        f.write("Final results of ")
        f.write(f"{timestamp}\n")
        f.write(f"{MODEL}\t{DATASET}\t{LOSS}\n")
        for k, v in vars(hp).items():
            if k == 'Batch_size' or k== 'alpha' or k== 'gamma':
                f.write(f"{k}: {v}\n")
        f.write("Tower Model: \n")
        f.write(f"Ending Epoch at {tower_end_epoch}\n")
        f.write(trainset_test_stable_results_s + '\n')
        f.write(validset_test_stable_results_s + '\n')
        f.write(testset_test_stable_results_s + '\n\n')

    writer.close()



def cat_train(SEED, DATASET, MODEL,  LOSS):
    '''set random seed'''
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    # cudnn.deterministic = True

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(DEVICE)
    '''init hyperparameters'''
    hp = hyperparameter(DATASET)

    '''load dataset from text file'''
    assert DATASET in ["DrugBank", "KIBA", "Davis", "Enzyme", "GPCRs", "ion_channel"]
    print("Train in " + DATASET)
    print("load data")
    dir_train = hp.dir_train
    with open(dir_train, "r") as f:
        train_data_list = f.read().strip().split('\n')
    print("train data load finished")
    print('Number of Train set: {}'.format(len(train_data_list)))
    '''shuffle data'''
    print("data shuffle")
    train_data_list = shuffle_dataset(train_data_list, SEED)

    dir_valid = hp.dir_valid
    with open(dir_valid, "r") as f:
        valid_data_list = f.read().strip().split('\n')
    print("valid data load finished")
    print('Number of Valid set: {}'.format(len(valid_data_list)))

    dir_test = hp.dir_test
    with open(dir_test, "r") as f:
        test_data_list = f.read().strip().split('\n')
    print("test data load finished")
    print('Number of Test set: {}'.format(len(test_data_list)))


    '''set loss function weight'''
    if DATASET == "Davis":
        weight_loss = torch.FloatTensor([0.3, 0.7]).to(DEVICE)
    elif DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.2, 0.8]).to(DEVICE)
    else:
        weight_loss = None

    # 加载 embeddingDataPrepare/embedding/tmp/index2id_drug_dict.json
    drug_rep = np.load(hp.embedding_path + '/best_user_embeddings.npy')  # [num_drugs, dim]
    pro_rep = np.load(hp.embedding_path + '/best_item_embeddings.npy')  # [num_proteins, dim]
    if isinstance(drug_rep, np.ndarray):
        drug_rep = torch.from_numpy(drug_rep).float().to(DEVICE)
    if isinstance(pro_rep, np.ndarray):
        pro_rep = torch.from_numpy(pro_rep).float().to(DEVICE)

    with open(hp.embedding_path + '/index2id_user_dict.json') as f:
        index2id_drug = json.load(f)
    id2index_drug = {v: int(k) for k, v in index2id_drug.items()}
    with open(hp.embedding_path + '/index2id_item_dict.json') as f:
        index2id_protein = json.load(f)
    id2index_protein = {v: int(k) for k, v in index2id_protein.items()}


    '''metrics'''
    Accuracy_List_stable, AUC_List_stable, AUPR_List_stable, Recall_List_stable, Precision_List_stable = [], [], [], [], []

    # train_dataset = CustomDataSet(train_data_list)
    # valid_dataset = CustomDataSet(valid_data_list)
    # test_dataset = CustomDataSet(test_data_list)
    train_dataset = CustomDataSet(train_data_list, id2index_drug, id2index_protein)
    valid_dataset = CustomDataSet(valid_data_list, id2index_drug, id2index_protein)
    test_dataset = CustomDataSet(test_data_list, id2index_drug, id2index_protein)
    train_size = len(train_dataset)

    train_dataset_loader = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
                                      collate_fn=collate_fn, drop_last=True)
    valid_dataset_loader = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                      collate_fn=collate_fn, drop_last=True)
    test_dataset_loader = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                     collate_fn=collate_fn, drop_last=True)

    model = GEDTI(hp).to(DEVICE)
    print(MODEL)
    """Initialize weights"""
    weight_p, bias_p = [], []
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]

    """create optimizer and scheduler"""
    optimizer = optim.AdamW(
        [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)

    scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False,
                                            step_size_up=train_size // hp.Batch_size)


    if LOSS == 'PolyLoss':
        Loss = PolyLoss(weight_loss=weight_loss,
                        DEVICE=DEVICE, epsilon=hp.loss_epsilon)
    elif LOSS == 'FocalLoss':
        Loss = FocalLoss(alpha=hp.alpha, gamma=hp.gamma)
    else:
        Loss = CELoss(weight_CE=weight_loss, DEVICE=DEVICE)

    """Output files"""
    save_path = hp.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    early_stopping = EarlyStopping(
        savepath=save_path, tower_patience=hp.Tower_patience, fusion_patience=hp.Fusion_patience, verbose=True, delta=0)
    stage = "Tower"

    writer = SummaryWriter(log_dir=os.path.join('runs', f'{MODEL}_{DATASET}_{LOSS}_{timestamp}'))
    """Start training."""
    print('Training...')
    tower_stage_ended = False
    for epoch in range(1, hp.Epoch + 1):
        if (early_stopping.tower_early_stop or epoch == hp.Tower_ending) and not tower_stage_ended:
            model.load_state_dict(torch.load(early_stopping.savepath + "/" + timestamp + '/valid_best_checkpoint_Tower.pth'))

            print('++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('start fusion stage...')
            stage = "Fusion"
            tower_end_epoch = epoch
            early_stopping.tower_early_stop = False
            tower_stage_ended = True
            early_stopping.best_score = -np.inf

        if early_stopping.fusion_early_stop:
            fusion_end_epoch = epoch
            break

        train_pbar = tqdm(
            enumerate(BackgroundGenerator(train_dataset_loader)),
            total=len(train_dataset_loader))

        """train"""
        train_losses_in_epoch = []
        model.train()
        for train_i, train_data in train_pbar:
            train_compounds, train_proteins, train_labels, drug_index, protein_index, _, _ = train_data
            train_compounds = train_compounds.to(DEVICE)
            train_proteins = train_proteins.to(DEVICE)
            train_labels = train_labels.to(DEVICE)
            batch_drug_rep = drug_rep[drug_index].to(DEVICE)
            batch_pro_rep = pro_rep[protein_index].to(DEVICE)

            optimizer.zero_grad()
            # predicted_interaction = model(train_compounds, train_proteins).squeeze(1)
            # predicted_interaction= model(train_compounds, train_proteins, batch_drug_rep, batch_pro_rep, epoch=epoch)

            pred_tower, pred_fusion = model(train_compounds, train_proteins, batch_drug_rep, batch_pro_rep, epoch, stage)

            if stage == "Tower":
                loss = Loss(pred_tower.squeeze(1), train_labels.float())
            else:
                loss = Loss(pred_fusion.squeeze(1), train_labels.float())


            train_losses_in_epoch.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()

        train_loss_a_epoch = np.average(train_losses_in_epoch)  # 一次epoch的平均训练loss

        """valid"""
        valid_pbar = tqdm(
            enumerate(BackgroundGenerator(valid_dataset_loader)),
            total=len(valid_dataset_loader))
        valid_losses_in_epoch = []
        model.eval()
        Y, P, S = [], [], []
        with torch.no_grad():
            for valid_i, valid_data in valid_pbar:

                valid_compounds, valid_proteins, valid_labels, drug_index, protein_index, _, _ = valid_data

                valid_compounds = valid_compounds.to(DEVICE)
                valid_proteins = valid_proteins.to(DEVICE)
                valid_labels = valid_labels.to(DEVICE)
                batch_drug_rep = drug_rep[drug_index].to(DEVICE)
                batch_pro_rep = pro_rep[protein_index].to(DEVICE)


                pred_tower, pred_fusion = model(valid_compounds, valid_proteins, batch_drug_rep, batch_pro_rep, epoch, stage)

                if stage == "Tower":
                    valid_loss = Loss(pred_tower.squeeze(1), valid_labels.float())
                    valid_scores = torch.sigmoid(pred_tower).detach().cpu().numpy()
                else:
                    valid_loss = Loss(pred_fusion.squeeze(1), valid_labels.float())
                    valid_scores = torch.sigmoid(pred_fusion).detach().cpu().numpy()

                valid_predictions = (valid_scores >= 0.5).astype(int)
                valid_labels = valid_labels.to('cpu').data.numpy()
                valid_losses_in_epoch.append(valid_loss.item())
                Y.extend(valid_labels)
                P.extend(valid_predictions)
                S.extend(valid_scores)

        Precision_dev = precision_score(Y, P)
        Reacll_dev = recall_score(Y, P)
        Accuracy_dev = accuracy_score(Y, P)
        F1_dev = f1_score(Y, P)
        AUC_dev = roc_auc_score(Y, S)
        tpr, fpr, _ = precision_recall_curve(Y, S)
        PRC_dev = auc(fpr, tpr)
        valid_loss_a_epoch = np.average(valid_losses_in_epoch)

        epoch_len = len(str(hp.Epoch))
        print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                     f'train_loss: {train_loss_a_epoch:.5f} ' +
                     f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                     f'valid_AUC: {AUC_dev:.5f} ' +
                     f'valid_PRC: {PRC_dev:.5f} ' +
                     f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                     f'valid_Precision: {Precision_dev:.5f} ' +
                     f'valid_Reacll: {Reacll_dev:.5f} '+
                     f'valid_F1_score: {F1_dev:.5f} ' )
        print(print_msg)
        writer.add_scalar('Loss/train', train_loss_a_epoch, epoch)
        writer.add_scalar('Loss/valid', valid_loss_a_epoch, epoch)
        writer.add_scalar('Metric/AUC', AUC_dev, epoch)
        writer.add_scalar('Metric/PRC', PRC_dev, epoch)
        writer.add_scalar('Metric/Accuracy', Accuracy_dev, epoch)
        writer.add_scalar('Metric/Precision', Precision_dev, epoch)
        writer.add_scalar('Metric/Recall', Reacll_dev, epoch)
        writer.add_scalar('Metric/F1_score', F1_dev, epoch)
        writer.add_scalar('Metric/LR', optimizer.param_groups[0]['lr'], epoch)

        '''save checkpoint and make decision when early stop'''

        # early_stopping(valid_loss_a_epoch, model, timestamp, stage)
        early_stopping(F1_dev, model, timestamp, stage)

    ##########################################
    '''load best checkpoint'''
    model.load_state_dict(torch.load(
        early_stopping.savepath + "/" + timestamp + '/valid_best_checkpoint_Tower.pth'))

    '''test model'''
    trainset_test_stable_results_s, _, _, _, _, _, _ = test_model(
        model, drug_rep, pro_rep, train_dataset_loader, save_path, timestamp, DATASET, Loss, DEVICE,  dataset_class="Train", stage = "Tower", FOLD_NUM=1)
    validset_test_stable_results_s, _, _, _, _, _, _ = test_model(
        model, drug_rep, pro_rep, valid_dataset_loader, save_path, timestamp, DATASET, Loss, DEVICE, dataset_class="Valid", stage = "Tower", FOLD_NUM=1)
    testset_test_stable_results_s, Accuracy_test, Precision_test, Recall_test, F1_test, AUC_test, PRC_test = test_model(
        model, drug_rep, pro_rep, test_dataset_loader, save_path, timestamp, DATASET, Loss, DEVICE, dataset_class="Test", stage = "Tower", FOLD_NUM=1)

    model.load_state_dict(torch.load(
        early_stopping.savepath + "/" + timestamp + '/valid_best_checkpoint_Fusion.pth'))
    trainset_test_stable_results_f, _, _, _, _, _, _ = test_model(
        model, drug_rep, pro_rep, train_dataset_loader, save_path, timestamp, DATASET, Loss, DEVICE,  dataset_class="Train", stage = "Fusion", FOLD_NUM=1)
    validset_test_stable_results_f, _, _, _, _, _, _ = test_model(
        model, drug_rep, pro_rep, valid_dataset_loader, save_path, timestamp, DATASET, Loss, DEVICE, dataset_class="Valid", stage = "Fusion", FOLD_NUM=1)
    testset_test_stable_results_f, Accuracy_test, Precision_test, Recall_test, F1_test, AUC_test, PRC_test = test_model(
        model, drug_rep, pro_rep, test_dataset_loader, save_path, timestamp, DATASET, Loss, DEVICE, dataset_class="Test", stage = "Fusion", FOLD_NUM=1)

    with open(save_path + '/' + "The_cat_results_of_whole_dataset.txt", 'a') as f:
        f.write("Final results of ")
        f.write(f"{timestamp}\n")
        f.write(f"{MODEL}\t{DATASET}\t{LOSS}\n")
        for k, v in vars(hp).items():
            if k == 'Batch_size' or k== 'alpha' or k== 'gamma':
                f.write(f"{k}: {v}\n")
        f.write("Tower Model: \n")
        f.write(f"Ending Epoch at {tower_end_epoch}\n")
        f.write(trainset_test_stable_results_s + '\n')
        f.write(validset_test_stable_results_s + '\n')
        f.write(testset_test_stable_results_s + '\n')
        f.write("Fusion Model: \n")
        f.write(f"Final Ending Epoch at {fusion_end_epoch}\n")
        f.write(trainset_test_stable_results_f + '\n')
        f.write(validset_test_stable_results_f + '\n')
        f.write(testset_test_stable_results_f + '\n\n')

    writer.close()

