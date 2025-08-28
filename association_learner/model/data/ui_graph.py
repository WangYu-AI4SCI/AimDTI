import os

import numpy as np
from collections import defaultdict
from data.data import Data
from data.graph import Graph
import scipy.sparse as sp
import json



class Interaction(Data, Graph):
    def __init__(self, conf, training, validation, test):
        Graph.__init__(self)
        Data.__init__(self, conf, training, validation, test)

        self.config = conf
        self.save_id_path = self.config['save_embedding']
        self.user = {} # id → index
        self.item = {}
        self.id2user = {} # index → id
        self.id2item = {}
        
        self.training_set_u = defaultdict(dict)  # 嵌套字典结构
        self.training_set_i = defaultdict(dict)
        self.training_set_u_neg = defaultdict(list)  # 负交互：user -> neg_item list

        self.validation_set = defaultdict(dict)
        self.validation_set_item = set()

        self.test_set = defaultdict(dict)
        self.test_set_item = set()

        self.__generate_set()
        self.user_num = len(self.training_set_u)
        self.item_num = len(self.training_set_i)
        self.ui_adj = self.__create_sparse_bipartite_adjacency()    # user-item 二分图的邻接矩阵
        self.norm_adj = self.normalize_graph_mat(self.ui_adj) 
        self.interaction_mat = self.__create_sparse_interaction_matrix() # user-item 交互矩阵

    def __generate_set(self):
        # 训练集
        for user, item, rating in self.training_data:
            # 分配 user_index
            if user not in self.user:
                user_index = len(self.user)
                self.user[user] = user_index
                self.id2user[user_index] = user
            # 分配 item_index
            if item not in self.item:
                item_index = len(self.item)
                self.item[item] = item_index
                self.id2item[item_index] = item
            # 构造训练集的 user-item 交互字典
            self.training_set_u[user][item] = rating  # 某个 user 交互的 item 集
            self.training_set_i[item][user] = rating  # 某个 item 交互的 user 集

        os.makedirs(self.save_id_path, exist_ok=True)
        # 写入文件 {index → id} 字典
        with open(self.save_id_path  + '/index2id_user_dict.json', 'w') as f:
            json.dump(self.id2user, f, indent=4)  
        with open(self.save_id_path  + '/index2id_item_dict.json', 'w') as f:
            json.dump(self.id2item, f, indent=4)     
        
        # 按照 index 排序的 id 列表
        index2id_user_list, index2id_item_list = [], []
        index2id_user_list = [self.id2user[i] for i in range(len(self.id2user))]
        index2id_item_list = [self.id2item[i] for i in range(len(self.id2item))]
        # 写入文件 
        with open(self.save_id_path  + '/index2id_user_list.json', 'w') as f:
            json.dump(index2id_user_list, f, indent=4)  
        with open(self.save_id_path  + '/index2id_item_list.json', 'w') as f:
            json.dump(index2id_item_list, f, indent=4) 

        # 拆分 0/1
        for user, item_dict in self.training_set_u.items():
            for item, rating in item_dict.items():
                if rating == 0:
                    self.training_set_u_neg[user].append(item)

        # 验证集
        for user, item, rating in self.validation_data:
            if user in self.user and item in self.item:
                self.validation_set[user][item] = rating
                self.validation_set_item.add(item)
        # 测试集    
        for user, item, rating in self.test_data:
            if user in self.user and item in self.item:
                self.test_set[user][item] = rating
                self.test_set_item.add(item)

    # 构建稀疏用户-物品二分图邻接矩阵（喜欢图）
    def __create_sparse_bipartite_adjacency(self, self_connection=False):
        n_nodes = self.user_num + self.item_num  # 构建一个大矩阵 user在前 item在后
        user_np = np.array([self.user[pair[0]] for pair in self.training_data])
        item_np = np.array([self.item[pair[1]] for pair in self.training_data]) + self.user_num
        ratings = np.array([pair[2] for pair in self.training_data]) # 有0 有1
        # ratings = np.ones_like(user_np, dtype=np.float32)
        tmp_adj = sp.csr_matrix((ratings, (user_np, item_np)), shape=(n_nodes, n_nodes), dtype=np.float32)
        adj_mat = tmp_adj + tmp_adj.T  # 对称化（无向图）
        if self_connection:
            adj_mat += sp.eye(n_nodes)
        return adj_mat

    # 把邻接矩阵转换成 Laplacian 矩阵，用于图卷积传播
    def convert_to_laplacian_mat(self, adj_mat):
        user_np_keep, item_np_keep = adj_mat.nonzero()
        ratings_keep = adj_mat.data
        tmp_adj = sp.csr_matrix((ratings_keep, (user_np_keep, item_np_keep + adj_mat.shape[0])),
                                shape=(adj_mat.shape[0] + adj_mat.shape[1], adj_mat.shape[0] + adj_mat.shape[1]),
                                dtype=np.float32)
        tmp_adj = tmp_adj + tmp_adj.T
        return self.normalize_graph_mat(tmp_adj)

    # 构造稀疏交互矩阵（用户在行、物品在列）（positive）
    def __create_sparse_interaction_matrix(self):
        row = np.array([self.user[pair[0]] for pair in self.training_data])
        col = np.array([self.item[pair[1]] for pair in self.training_data])
        entries = np.array([pair[2] for pair in self.training_data])
        # entries = np.ones(len(row), dtype=np.float32)
        return sp.csr_matrix((entries, (row, col)), shape=(self.user_num, self.item_num), dtype=np.float32)

    def get_user_id(self, u):
        return self.user.get(u)

    def get_item_id(self, i):
        return self.item.get(i)

    def training_size(self):
        return len(self.user), len(self.item), len(self.training_data)

    def validation_size(self):
        return len(self.validation_set), len(self.validation_set_item), len(self.validation_data)

    def test_size(self):
        return len(self.test_set), len(self.test_set_item), len(self.test_data)

    def contain(self, u, i):
        return u in self.user and i in self.training_set_u[u]

    def contain_user(self, u):
        return u in self.user

    def contain_item(self, i):
        return i in self.item

    def user_rated(self, u):
        return list(self.training_set_u[u].keys()), list(self.training_set_u[u].values())

    def item_rated(self, i):
        return list(self.training_set_i[i].keys()), list(self.training_set_i[i].values())

    def row(self, u):
        k, v = self.user_rated(self.id2user[u])
        vec = np.zeros(self.item_num, dtype=np.float32)
        for item, rating in zip(k, v):
            vec[self.item[item]] = rating
        return vec

    def col(self, i):
        k, v = self.item_rated(self.id2item[i])
        vec = np.zeros(self.user_num, dtype=np.float32)
        for user, rating in zip(k, v):
            vec[self.user[user]] = rating
        return vec

    def matrix(self):
        m = np.zeros((self.user_num, self.item_num), dtype=np.float32)
        for u, u_id in self.user.items():
            vec = np.zeros(self.item_num, dtype=np.float32)
            k, v = self.user_rated(u)
            for item, rating in zip(k, v):
                vec[self.item[item]] = rating
            m[u_id] = vec
        return m
