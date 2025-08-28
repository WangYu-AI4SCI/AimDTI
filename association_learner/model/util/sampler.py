from random import shuffle,randint,choice,sample
import numpy as np


def next_batch_binary(data,batch_size,n_negs=20):
    training_data = data.training_data
    shuffle(training_data)
    ptr = 0
    data_size = len(training_data)
    while ptr < data_size:
        if ptr + batch_size < data_size:
            batch_end = ptr + batch_size
        else:
            batch_end = data_size
        users = [training_data[idx][0] for idx in range(ptr, batch_end)]
        items = [training_data[idx][1] for idx in range(ptr, batch_end)]
        ratings = [training_data[idx][2] for idx in range(ptr, batch_end)]
        ptr = batch_end
        u_idx, i_idx, labels = [], [], []
        for i, user in enumerate(users):
            label = ratings[i]
            if(label == 1): # 每遇到一个正样本，为这个drug找负样本
                u_idx.append(data.user[user])
                i_idx.append(data.item[items[i]])
                labels.append(1)
                if user not in data.training_set_u_neg or not data.training_set_u_neg[user]:
                    continue  # 跳过该drug，避免空采样
                for m in range(n_negs):
                    u_idx.append(data.user[user])
                    neg_item = choice(data.training_set_u_neg[user])
                    i_idx.append(data.item[neg_item])
                    labels.append(0)
        yield  u_idx, i_idx, labels


def next_batch_binary_test(data, batch_size, dataset):
    ptr = 0
    if dataset == 'test':
    # for testing model
        datatest = data.test_data
    elif dataset == 'validation':
        datatest = data.validation_data
    else:
        datatest = data.training_data

    shuffle(datatest)
    data_size = len(datatest)

    while ptr < data_size:
        batch_end = min(ptr + batch_size, data_size)
        users = [datatest[idx][0] for idx in range(ptr, batch_end)]
        items = [datatest[idx][1] for idx in range(ptr, batch_end)]
        labels = [datatest[idx][2] for idx in range(ptr, batch_end)]  # 0 or 1
        # print("user-0",users[0])
        ptr = batch_end
        u_idx, i_idx = [], []
        for i, user in enumerate(users):
            if items[i] in data.item:
                u_idx.append(data.user[user])
                i_idx.append(data.item[items[i]])
                # labels.append(ratings[i])
        yield users, items, u_idx, i_idx, labels

        # # for drugbank
        # batch_end = min(ptr + batch_size, data_size)
        # users = [datatest[idx][0] for idx in range(ptr, batch_end)]
        # items = [datatest[idx][1] for idx in range(ptr, batch_end)]
        # labels = [datatest[idx][2] for idx in range(ptr, batch_end)]  # 0 or 1
        # # print("user-0",users[0])
        # ptr = batch_end
        # filtered_users, filtered_items, u_idx, i_idx, filtered_labels = [], [], [], [], []
        # for i, user in enumerate(users):
        #     if items[i] in data.item:
        #         filtered_users.append(users[i])
        #         filtered_items.append(items[i])
        #         u_idx.append(data.user[user])
        #         i_idx.append(data.item[items[i]])
        #         filtered_labels.append(labels[i])
        # yield filtered_users, filtered_items, u_idx, i_idx, filtered_labels




