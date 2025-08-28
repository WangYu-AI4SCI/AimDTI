import torch
import torch.nn.functional as F
import torch.nn as nn


def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.mul(user_emb, pos_item_emb).sum(dim=1)
    neg_score = torch.mul(user_emb, neg_item_emb).sum(dim=1)
    loss = -torch.log(10e-6 + torch.sigmoid(pos_score - neg_score))
    return torch.mean(loss)

def triplet_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = ((user_emb-pos_item_emb)**2).sum(dim=1)
    neg_score = ((user_emb-neg_item_emb)**2).sum(dim=1)
    loss = F.relu(pos_score-neg_score+0.5)
    return torch.mean(loss)

def l2_reg_loss(reg, *args):
    emb_loss = 0
    for emb in args:
        emb_loss += torch.norm(emb, p=2)/emb.shape[0]
    return emb_loss * reg

# 0.8 1.0
def focal_loss(user_emb, item_emb, labels, alpha=0.75, gamma=1.0):
    logits = torch.sum(user_emb * item_emb, dim=1)  # 点积作为预测
    probs = torch.sigmoid(logits)
    pt = probs * labels + (1 - probs) * (1 - labels)
    alpha_t = alpha * labels + (1 - alpha) * (1 - labels)

    loss = -alpha_t * (1 - pt) ** gamma * torch.log(pt + 1e-8)
    return loss.mean()

def batch_softmax_loss(user_emb, item_emb, temperature):
    user_emb, item_emb = F.normalize(user_emb, dim=1), F.normalize(item_emb, dim=1)
    pos_score = (user_emb * item_emb).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(user_emb, item_emb.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    loss = -torch.log(pos_score / ttl_score+10e-6)
    return torch.mean(loss)


# def InfoNCE(view1, view2, temperature: float, b_cos: bool = True):
#     """
#     Args:
#         view1: (torch.Tensor - N x D)
#         view2: (torch.Tensor - N x D)
#         temperature: float
#         b_cos (bool)
#
#     Return: Average InfoNCE Loss
#     """
#     if b_cos:
#         view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
#
#     pos_score = (view1 @ view2.T) / temperature
#     score = torch.diag(F.log_softmax(pos_score, dim=1))
#     return -score.mean()

# def InfoNCE(view1, view2, temperature: float = 0.2):
#     view1 = F.normalize(view1, dim=1)
#     view2 = F.normalize(view2, dim=1)
#
#     logits = view1 @ view2.T / temperature  # [N, N]
#     labels = torch.arange(logits.shape[0], device=logits.device)
#     loss_1 = F.cross_entropy(logits, labels)
#     loss_2 = F.cross_entropy(logits.T, labels)
#     return (loss_1 + loss_2) / 2

def InfoNCE(view1, view2, temperature=0.2, b_cos=True):
    if b_cos:
        view1 = F.normalize(view1, dim=1)
        view2 = F.normalize(view2, dim=1)
    sim_matrix = torch.matmul(view1, view2.T) / temperature

    # 避免 NaN：排除对角线自己，确保 log_softmax 是有效的
    log_prob = F.log_softmax(sim_matrix, dim=1)
    positive = torch.diag(log_prob)
    return -positive.mean()

def info_nce_loss_cosine(view1, view2, temperature=0.2):
    """
    Cosine InfoNCE loss for positive pairs (view1, view2).
    view1: [N, D]
    view2: [N, D]
    """
    view1 = F.normalize(view1, dim=1)
    view2 = F.normalize(view2, dim=1)
    N = view1.size(0)

    # [2N, D]
    embeddings = torch.cat([view1, view2], dim=0)  # positive + positive
    # [2N, 2N]
    sim_matrix = torch.matmul(embeddings, embeddings.T)  # cosine sim

    # Create mask to remove similarity with itself
    mask = torch.eye(2 * N, dtype=torch.bool).to(embeddings.device)
    sim_matrix = sim_matrix[~mask].view(2 * N, -1)

    # positive similarities (i-th and (i+N)-th are pairs)
    pos_sim = torch.sum(view1 * view2, dim=-1) / temperature
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    # logits: [2N]
    logits = torch.cat([pos_sim.unsqueeze(1), sim_matrix / temperature], dim=1)

    labels = torch.zeros(2 * N, dtype=torch.long).to(view1.device)  # pos index = 0
    loss = F.cross_entropy(logits, labels)

    return loss





#this version is from recbole
def info_nce(z_i, z_j, temp, batch_size, sim='dot'):
    """
    We do not sample negative examples explicitly.
    Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
    """
    def mask_correlated_samples(batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    N = 2 * batch_size

    z = torch.cat((z_i, z_j), dim=0)

    if sim == 'cos':
        sim = nn.functional.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / temp
    elif sim == 'dot':
        sim = torch.mm(z, z.T) / temp

    sim_i_j = torch.diag(sim, batch_size)
    sim_j_i = torch.diag(sim, -batch_size)

    positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)

    mask = mask_correlated_samples(batch_size)

    negative_samples = sim[mask].reshape(N, -1)

    labels = torch.zeros(N).to(positive_samples.device).long()
    logits = torch.cat((positive_samples, negative_samples), dim=1)
    return F.cross_entropy(logits, labels)


def kl_divergence(p_logit, q_logit):
    p = F.softmax(p_logit, dim=-1)
    kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1) - F.log_softmax(q_logit, dim=-1)), 1)
    return torch.mean(kl)

