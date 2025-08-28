# -*- coding:utf-8 -*-

import json
import os

import torch
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, f1_score)
from collections import defaultdict


def test_precess(MODEL, drug_rep, pro_rep, pbar, LOSS, DEVICE, stage, FOLD_NUM):
    if isinstance(MODEL, list):
        for item in MODEL:
            item.eval()
    else:
        MODEL.eval()
    test_losses = []
    Y, P, S ,Pairs = [], [], [], []
    with torch.no_grad():
        for i, data in pbar:
            '''data preparation '''
            compounds, proteins, labels, drug_index, protein_index, drug_ids, protein_ids= data
            compounds = compounds.to(DEVICE)
            proteins = proteins.to(DEVICE)
            labels = labels.to(DEVICE)
            batch_drug_rep = drug_rep[drug_index].to(DEVICE)
            batch_pro_rep = pro_rep[protein_index].to(DEVICE)

            if isinstance(MODEL, list):
                pred_fusion = torch.zeros(2).to(DEVICE)
                for i in range(len(MODEL)):
                    pred_fusion = pred_fusion + \
                        MODEL[i](compounds, proteins)
                pred_fusion = pred_fusion / FOLD_NUM
            else:
                pred_struct, pred_fusion = MODEL(compounds, proteins, batch_drug_rep, batch_pro_rep, epoch=1, stage=stage)

                if stage == "Tower":
                    loss = LOSS(pred_struct.squeeze(1), labels.float())
                    predicted_scores = torch.sigmoid(pred_struct).detach().cpu().numpy()
                else:
                    loss = LOSS(pred_fusion.squeeze(1), labels.float())
                    predicted_scores = torch.sigmoid(pred_fusion).detach().cpu().numpy()


            correct_labels = labels.to('cpu').data.numpy()
            predicted_labels = (predicted_scores >= 0.5).astype(int)


            pairs = [f"{drug_id} {protein_id}" for drug_id, protein_id in zip(drug_ids, protein_ids)]

            Pairs.extend(pairs)
            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())

    pairs_dict = {}
    for i in range(len(Pairs)):
        pairs_dict[str(Pairs[i])] = {
            "label": int(Y[i]),
            "prediction": int(P[i]),
            "score": float(S[i])
        }

    Precision = precision_score(Y, P)
    Recall = recall_score(Y, P)
    F1_score = f1_score(Y, P)
    AUC = roc_auc_score(Y, S)
    tpr, fpr, _ = precision_recall_curve(Y, S)
    PRC = auc(fpr, tpr)
    Accuracy = accuracy_score(Y, P)
    test_loss = np.average(test_losses)
    return pairs_dict, test_loss, Accuracy, Precision, Recall, F1_score, AUC, PRC


def test_model(MODEL, drug_rep, pro_rep, dataset_loader, save_path, timestamp, DATASET, LOSS, DEVICE, dataset_class="Train", stage = "Tower",save=False, FOLD_NUM=1):
    test_pbar = tqdm(
        enumerate(
            BackgroundGenerator(dataset_loader)),
        total=len(dataset_loader))
    pairs_dict, loss_test, Accuracy_test, Precision_test, Recall_test, F1_test, AUC_test, PRC_test = test_precess(
        MODEL, drug_rep, pro_rep, test_pbar, LOSS, DEVICE, stage, FOLD_NUM)

    if save:
        if FOLD_NUM == 1:
            filepath = save_path + "/" + timestamp + \
                "/{}_{}_prediction.json".format(DATASET, dataset_class)
        else:
            filepath = save_path + "/" + timestamp + \
                "/{}_{}_ensemble_prediction.json".format(DATASET, dataset_class)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(pairs_dict, f, indent=4)

    results = '{}: Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};F1_score:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
        .format(dataset_class, loss_test, Accuracy_test, Precision_test, Recall_test, F1_test, AUC_test, PRC_test)
    print(results)
    return results, Accuracy_test, Precision_test, Recall_test, F1_test, AUC_test, PRC_test


# # -*- coding:utf-8 -*-
# '''
# Author: MrZQAQ
# Date: 2022-03-29 14:00
# LastEditTime: 2022-11-23 15:32
# LastEditors: MrZQAQ
# Description: Test model
# FilePath: /MCANet/utils/TestModel.py
# '''
# import json
# import os
#
# import torch
# import numpy as np
# import torch.nn.functional as F
# from tqdm import tqdm
# from prefetch_generator import BackgroundGenerator
# from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
#                              precision_score, recall_score, roc_auc_score, f1_score)
# from collections import defaultdict
#
#
# def test_precess(MODEL, pbar, LOSS, DEVICE, FOLD_NUM):
#     if isinstance(MODEL, list):
#         for item in MODEL:
#             item.eval()
#     else:
#         MODEL.eval()
#     test_losses = []
#     Y, P, S ,Pairs = [], [], [], []
#     with torch.no_grad():
#         for i, data in pbar:
#             '''data preparation '''
#             compounds, proteins, labels, drug_ids, protein_ids = data
#             compounds = compounds.to(DEVICE)
#             proteins = proteins.to(DEVICE)
#             labels = labels.to(DEVICE)
#
#             if isinstance(MODEL, list):
#                 predicted_scores = torch.zeros(2).to(DEVICE)
#                 for i in range(len(MODEL)):
#                     predicted_scores = predicted_scores + \
#                         MODEL[i](compounds, proteins)
#                 predicted_scores = predicted_scores / FOLD_NUM
#             else:
#                 predicted_scores = MODEL(compounds, proteins)
#             loss = LOSS(predicted_scores.squeeze(1), labels.float())
#             correct_labels = labels.to('cpu').data.numpy()
#             predicted_scores = torch.sigmoid(predicted_scores).detach().cpu().numpy()
#             predicted_labels = (predicted_scores >= 0.5).astype(int)
#
#
#             pairs = [f"{drug_id} {protein_id}" for drug_id, protein_id in zip(drug_ids, protein_ids)]
#
#             Pairs.extend(pairs)
#             Y.extend(correct_labels)
#             P.extend(predicted_labels)
#             S.extend(predicted_scores)
#             test_losses.append(loss.item())
#
#     pairs_dict = {}
#     for i in range(len(Pairs)):
#         pairs_dict[str(Pairs[i])] = {
#             "label": int(Y[i]),
#             "prediction": int(P[i]),
#             "score": float(S[i])
#         }
#
#     Precision = precision_score(Y, P)
#     Recall = recall_score(Y, P)
#     F1_score = f1_score(Y, P)
#     AUC = roc_auc_score(Y, S)
#     tpr, fpr, _ = precision_recall_curve(Y, S)
#     PRC = auc(fpr, tpr)
#     Accuracy = accuracy_score(Y, P)
#     test_loss = np.average(test_losses)
#     return pairs_dict, test_loss, Accuracy, Precision, Recall, F1_score, AUC, PRC
#
#
# def test_model(MODEL, dataset_loader, save_path, timestamp, DATASET, LOSS, DEVICE, dataset_class="Train", save=True, FOLD_NUM=1):
#     test_pbar = tqdm(
#         enumerate(
#             BackgroundGenerator(dataset_loader)),
#         total=len(dataset_loader))
#     pairs_dict, loss_test, Accuracy_test, Precision_test, Recall_test, F1_test, AUC_test, PRC_test = test_precess(
#         MODEL, test_pbar, LOSS, DEVICE, FOLD_NUM)
#
#     if save:
#         if FOLD_NUM == 1:
#             filepath = save_path + "/" + timestamp + \
#                 "/{}_{}_prediction.json".format(DATASET, dataset_class)
#         else:
#             filepath = save_path + "/" + timestamp + \
#                 "/{}_{}_ensemble_prediction.json".format(DATASET, dataset_class)
#         os.makedirs(os.path.dirname(filepath), exist_ok=True)
#         with open(filepath, 'w') as f:
#             json.dump(pairs_dict, f, indent=4)
#
#     results = '{}: Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};F1_score:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
#         .format(dataset_class, loss_test, Accuracy_test, Precision_test, Recall_test, F1_test, AUC_test, PRC_test)
#     print(results)
#     return results, Accuracy_test, Precision_test, Recall_test, F1_test, AUC_test, PRC_test
