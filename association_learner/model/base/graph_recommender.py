import json
import datetime
import os

import numpy as np
from base.recommender import Recommender
from data.ui_graph import Interaction
from util.algorithm import find_k_largest
from data.loader import FileIO
from os.path import abspath
from util.evaluation import ranking_evaluation
from util.sampler import  next_batch_binary_test
from sklearn.metrics import accuracy_score, auc, precision_recall_curve,precision_score, recall_score, roc_auc_score,f1_score

class GraphRecommender(Recommender):
    def __init__(self, conf, training_set, validation_set, test_set, **kwargs):
        super(GraphRecommender, self).__init__(conf, training_set, validation_set, test_set, **kwargs)
        self.data = Interaction(conf, training_set, validation_set, test_set)

        self.topN = [int(num) for num in self.ranking]
        self.max_N = max(self.topN)
        self.counter = 0
        self.early_stop = False
        self.bestPerformance = {
            'epoch': 0,
            'metrics': {
                'loss': 5.0,
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'auc': 0.0,
                'prc': 0.0
            }
        }

    def print_model_info(self):
        super(GraphRecommender, self).print_model_info()
        # print dataset statistics
        # print(self.data.training_data)
        print(f'Training Set Size: (user number: {self.data.training_size()[0]}, '
              f'item number: {self.data.training_size()[1]}, '
              f'interaction number: {self.data.training_size()[2]})')
        print(f'Test Set Size: (user number: {self.data.test_size()[0]}, '
              f'item number: {self.data.test_size()[1]}, '
              f'interaction number: {self.data.test_size()[2]})')
        print('=' * 80)

    def build(self):
        pass

    def train(self):
        pass

    def predict(self, u):
        pass

    def test(self, dataset):
        Y, P, S, Pairs = [], [], [], []
        test_losses = []
        for n, batch in enumerate(next_batch_binary_test(self.data, self.batch_size, dataset)):
            users, items, user_idx, item_idx, labels = batch


            scores, loss = self.predict(user_idx, item_idx, labels)
            # preds = (scores >= 0.5).long()
            preds = (scores >= 0.5).astype(int)

            pairs = [f"{drug_id} {protein_id}" for drug_id, protein_id in zip(users, items)]

            Pairs.extend(pairs)
            Y.extend(labels)
            P.extend(preds.tolist())
            S.extend(scores.tolist())
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
        F1 = f1_score(Y, P)
        Accuracy = accuracy_score(Y, P)
        AUC = roc_auc_score(Y, S)
        tpr, fpr, _ = precision_recall_curve(Y, S)
        PRC = auc(fpr, tpr)
        test_loss = np.average(test_losses)
        # 保存为结果字典
        return pairs_dict, test_loss, Accuracy, Precision, Recall, F1, AUC, PRC


    def evaluate(self):
        pairs_dict_train, loss_train, Accuracy_train, Precision_train, Recall_train, F1_train, AUC_train, PRC_train = self.test('train')
        pairs_dict_valid, loss_valid, Accuracy_valid, Precision_valid, Recall_valid, F1_valid, AUC_valid, PRC_valid = self.test('validation')
        pairs_dict_test, loss_test, Accuracy_test, Precision_test, Recall_test, F1_test, AUC_test, PRC_test = self.test('test')

        timestamp = self.timestamp
        DataName = self.DataName
        save_results = self.save_dir + "/" + timestamp
        os.makedirs(save_results, exist_ok=True)

        filepath = save_results  + \
                   "/{}_{}_prediction.json".format(DataName, 'train')
        with open(filepath, 'w') as f:
            json.dump(pairs_dict_train, f, indent=4)
        results_train = '{}: Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};F1:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
            .format('train', loss_train, Accuracy_train, Precision_train, Recall_train, F1_train, AUC_train, PRC_train)
        print(results_train)

        filepath = save_results + \
                   "/{}_{}_prediction.json".format(DataName, 'validation')
        with open(filepath, 'w') as f:
            json.dump(pairs_dict_valid, f, indent=4)
        results_valid = '{}: Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};F1:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
            .format('valid', loss_valid, Accuracy_valid, Precision_valid, Recall_valid, F1_valid, AUC_valid, PRC_valid)
        print(results_valid)

        filepath = save_results  + \
                   "/{}_{}_prediction.json".format(DataName, 'test')
        with open(filepath, 'w') as f:
            json.dump(pairs_dict_test, f, indent=4)
        results_test = '{}: Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};F1:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
            .format('test', loss_test, Accuracy_test, Precision_test, Recall_test, F1_test, AUC_test, PRC_test)
        print(results_test)

        return results_train, results_valid, results_test


    def fast_evaluation(self, epoch, dataset):

        print('Fast Evaluating the model...')
        _, loss_dev, Accuracy_dev, Precision_dev, Recall_dev, F1_dev, AUC_dev, PRC_dev = self.test(dataset)

        res_dict = {
            "loss": loss_dev,
            "accuracy": Accuracy_dev,
            "precision": Precision_dev,
            "recall": Recall_dev,
            "auc": AUC_dev,
            "prc": PRC_dev,
        }

        results = '{}: Loss:{:.5f};Accuracy:{:.5f};Precision:{:.5f};Recall:{:.5f};F1:{:.5f};AUC:{:.5f};PRC:{:.5f}.' \
            .format(dataset, loss_dev, Accuracy_dev, Precision_dev, Recall_dev, F1_dev, AUC_dev, PRC_dev)
        print(results)

        if self.bestPerformance is None:
            self.bestPerformance = {
                'epoch': epoch + 1,
                'metrics': res_dict
            }
            self.save()
            print(f"初始化 bestPerformance: epoch {epoch + 1}")
        else:
            # old_auc = self.bestPerformance['metrics'].get('auc', 0)
            # new_auc = res_dict.get('auc', 0)
            old_loss = self.bestPerformance['metrics'].get('loss', 0)
            new_loss = res_dict.get('loss', 0)
            if new_loss < old_loss:
                self.bestPerformance = {
                    'epoch': epoch + 1,
                    'metrics': res_dict
                }
                self.save()
                print(f"新的最佳模型: valid loss 从 {old_loss:.4f} 提升到 {new_loss:.4f}, 已保存。")
                # print(f"新的最佳模型: valid auc 从 {old_loss:.4f} 提升到 {new_loss:.4f}, 已保存。")
                self.counter = 0
            else:
                self.counter += 1
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True

        return loss_dev
