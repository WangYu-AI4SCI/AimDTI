import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from base.graph_recommender import GraphRecommender
from util.sampler import next_batch_binary
from base.torch_interface import TorchGraphInterface
from util.loss_torch import bpr_loss,l2_reg_loss, focal_loss, InfoNCE, info_nce_loss_cosine
from torch.utils.tensorboard import SummaryWriter
import random
from tqdm import tqdm


class LightGCN(GraphRecommender):
    def __init__(self, conf, training_set, validation_set, test_set):
        super(LightGCN, self).__init__(conf, training_set, validation_set, test_set)
        self.args = self.config['LightGCN']
        self.n_layers = int(self.args['n_layer'])
        self.eps = float(self.args['eps'])
        self.model = LGCN_Encoder(self.data, self.emb_size, self.n_layers, self.eps)
        self.cl_rate = float(self.args['lambda'])


    def train(self):
        writer = SummaryWriter(log_dir='runs/lightGCN')
        model = self.model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lRate)

        for epoch in range(self.maxEpoch):
            train_loss_a_epoch = []
            for n, batch in enumerate(next_batch_binary(self.data, self.batch_size)):
                user_idx, item_idx, labels = batch
                user_idx = torch.tensor(user_idx, dtype=torch.long).cuda()
                item_idx = torch.tensor(item_idx, dtype=torch.long).cuda()
                labels = torch.tensor(labels).float().cuda()

                rec_user_emb, rec_item_emb = model()
                user_emb = rec_user_emb[user_idx]
                item_emb = rec_item_emb[item_idx]

                loss = focal_loss(user_emb, item_emb, labels)
                # L2正则
                reg_loss = l2_reg_loss(self.reg,
                                       model.embedding_dict['user_emb'][user_idx],
                                       model.embedding_dict['item_emb'][item_idx]) / self.batch_size

                triplets = self.construct_triplets(user_idx, item_idx, labels)

                if len(triplets) > 0:
                    u_list, pos_list, neg_list = zip(*triplets)
                    u_list = torch.tensor(u_list, dtype=torch.long).cuda()
                    pos_list = torch.tensor(pos_list, dtype=torch.long).cuda()
                    neg_list = torch.tensor(neg_list, dtype=torch.long).cuda()

                    rec_user_emb, rec_item_emb = model()
                    u_emb = rec_user_emb[u_list]
                    pos_emb = rec_item_emb[pos_list]
                    neg_emb = rec_item_emb[neg_list]

                    triplet_loss =  self.cl_rate * self.cal_triplet_loss(u_emb, pos_emb, neg_emb)
                    # triplet_loss =  self.cal_triplet_loss(u_emb, pos_emb, neg_emb)

                else:
                    triplet_loss = 0

                # cl_loss = self.cl_rate * self.cal_supcon_loss(user_emb, item_emb, labels)
                #
                if epoch > 20:  # 例如 hp.cl_warmup_epoch = 20
                    batch_loss = loss + reg_loss + triplet_loss
                else:
                    batch_loss = loss + reg_loss  # 不加 cl_loss

                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                train_loss_a_epoch.append(batch_loss.item())
                writer.add_scalar('Loss/train', batch_loss.item(), epoch)
                writer.add_scalar('Loss/focal', loss.item(), epoch)
                writer.add_scalar('Loss/cl', triplet_loss.item(), epoch)


            print(f"Training loss at epoch {epoch}: {np.average(train_loss_a_epoch):.4f}")
            with torch.no_grad():
                self.user_emb, self.item_emb = model()

            valid_loss_a_epoch = self.fast_evaluation(epoch, 'validation')
            writer.add_scalar('Loss/valid', valid_loss_a_epoch, epoch)
            if self.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break

        self.user_emb, self.item_emb = self.best_user_emb, self.best_item_emb

    def cal_supcon_loss(self, user_emb, item_emb, labels, temperature=0.2):
        """
        基于监督信号的对比损失，正样本为 label=1 的 (u,i) 对之间的 embedding
        """
        pos_mask = labels == 1
        if pos_mask.sum() < 2:
            return torch.tensor(0.0, device=user_emb.device)  # 无法构建正对

        # 只用正样本对参与 supcon loss
        z = F.normalize(user_emb[pos_mask] * item_emb[pos_mask], dim=1)
        sim_matrix = torch.mm(z, z.t()) / temperature

        # 去除对角线（自身）
        mask = torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
        sim_matrix = sim_matrix[~mask].view(sim_matrix.size(0), -1)

        positives = torch.exp(sim_matrix)
        loss = -torch.log(positives / positives.sum(dim=1, keepdim=True))
        return loss.mean()

    def construct_triplets(self, user_idx, item_idx, labels):
        pos_triplets = []
        neg_dict = {}

        for u, i, l in zip(user_idx, item_idx, labels):
            if l == 1:  # 正样本
                pos_triplets.append((u.item(), i.item()))
            else:
                neg_dict.setdefault(u.item(), []).append(i.item())

        triplets = []
        for u, i_pos in pos_triplets:
            if u in neg_dict and len(neg_dict[u]) > 0:
                i_neg = random.choice(neg_dict[u])
                triplets.append((u, i_pos, i_neg))

        return triplets  # [(u, i_pos, i_neg), ...]

    def cal_triplet_loss(self, user_emb, pos_item_emb, neg_item_emb, margin=2.0):
        pos_dist = F.pairwise_distance(user_emb, pos_item_emb)
        neg_dist = F.pairwise_distance(user_emb, neg_item_emb)
        loss = F.relu(pos_dist - neg_dist + margin)
        return loss.mean()

    def save(self):
        with torch.no_grad():
            self.best_user_emb, self.best_item_emb = self.model.forward()
            
        # 将最佳模型的 node embedding 保存为 numpy 文件
        user_emb_np = self.best_user_emb.cpu().numpy()
        item_emb_np = self.best_item_emb.cpu().numpy()
        np.save(self.save_embedding + '/best_user_embeddings.npy', user_emb_np)
        np.save(self.save_embedding + '/best_item_embeddings.npy', item_emb_np)



    def predict(self, user_idx, item_idx, labels):
        user_emb = self.user_emb[user_idx]
        item_emb = self.item_emb[item_idx]
        logits = (user_emb * item_emb).sum(dim=1)  # 点积作为评分
        preds = torch.sigmoid(logits)
        labels = torch.tensor(labels).float().cuda()
        loss = focal_loss(user_emb, item_emb, labels)
        batch_loss = loss

        return preds.cpu().numpy(), batch_loss



class LGCN_Encoder(nn.Module):
    def __init__(self, data, emb_size, n_layers, eps):
        super(LGCN_Encoder, self).__init__()
        self.data = data
        self.latent_size = emb_size
        self.layers = n_layers
        self.norm_adj = data.norm_adj  # data.norm_adj 是预先计算好的归一化邻接矩阵
        self.sparse_norm_adj = TorchGraphInterface.convert_sparse_mat_to_tensor(self.norm_adj).cuda()
        self.embedding_dict = self._init_model()
        self.eps = eps
        

    def _init_model(self):
        initializer = nn.init.xavier_uniform_
        embedding_dict = nn.ParameterDict({
            'user_emb': nn.Parameter(initializer(torch.empty(self.data.user_num, self.latent_size))), # user 嵌入矩阵
            'item_emb': nn.Parameter(initializer(torch.empty(self.data.item_num, self.latent_size))), # item 嵌入矩阵
        })
        return embedding_dict
    
 
    def forward(self, perturbed=False):
        ego_embeddings = torch.cat([self.embedding_dict['user_emb'], self.embedding_dict['item_emb']], 0)

        all_embeddings = [ego_embeddings]
        for k in range(self.layers):
            ego_embeddings = torch.sparse.mm(self.sparse_norm_adj, ego_embeddings)
            if perturbed:
                random_noise = torch.rand_like(ego_embeddings).cuda()
                ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, dim=-1) * self.eps
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        user_all_embeddings = all_embeddings[:self.data.user_num]
        item_all_embeddings = all_embeddings[self.data.user_num:]
        return user_all_embeddings, item_all_embeddings


