import torch
import torch.nn as nn
import torch.nn.functional as F

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean'else loss.sum() if reduction=='sum'else loss

def linear_combination(x, y, epsilon):
    return epsilon*x + (1-epsilon)*y

class LabelSmoothCELoss(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean', ignore_index=255):
        super(LabelSmoothCELoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, preds, labels):
        n = preds.size()[1]
        log_preds = F.log_softmax(preds, dim=1)
        loss = reduce_loss(-log_preds.sum(dim=1), self.reduction)
        nll = F.nll_loss(log_preds, labels, reduction=self.reduction, ignore_index=self.ignore_index)
        return linear_combination(loss/n, nll, self.epsilon)

