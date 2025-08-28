# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


class PolyLoss(nn.Module):
    def __init__(self, weight_loss, DEVICE, epsilon=1.0):
        super(PolyLoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_loss, reduction='none')
        self.epsilon = epsilon
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        one_hot = torch.zeros((16, 2), device=self.DEVICE).scatter_(
            1, torch.unsqueeze(labels, dim=-1), 1)
        pt = torch.sum(one_hot * F.softmax(predicted, dim=1), dim=-1)
        ce = self.CELoss(predicted, labels)
        poly1 = ce + self.epsilon * (1-pt)
        return torch.mean(poly1)


class CELoss(nn.Module):
    def __init__(self, weight_CE, DEVICE):
        super(CELoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_CE)
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        return self.CELoss(predicted, labels)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        inputs: raw logits, shape [B]
        targets: 0 or 1, shape [B]
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)

        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_term = (1 - pt) ** self.gamma

        loss = alpha_t * focal_term * BCE_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
