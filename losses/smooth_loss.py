import torch
import torch.nn as nn
import torch.nn.functional as F

def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean'else loss.sum() if reduction=='sum'else loss

def linear_combination(x, y, epsilon):
    return epsilon*x + (1-epsilon)*y

class LabelSmoothCELoss_b(nn.Module):
    def __init__(self, epsilon=0.1, reduction='mean', ignore_index=255):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, preds, labels):
        n = preds.size()[1]
        log_preds = F.log_softmax(preds, dim=1)
        loss = reduce_loss(-log_preds.sum(dim=1), self.reduction)
        nll = F.nll_loss(log_preds, labels, reduction=self.reduction, ignore_index=self.ignore_index)
        return linear_combination(loss/n, nll, self.epsilon)


class LabelSmoothCELoss(nn.Module):
    def __init__(self, eps=0.1, reduction='mean', ignore_index=255):
        super().__init__()
        self.eps = eps
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, output, target):
        c = output.size()[1]
        log_preds = F.log_softmax(output, dim=1)

        loss = log_preds.permute(0, 2, 3, 1)
        loss = -loss[target != self.ignore_index].sum(dim=1)

        if self.reduction=='sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        nll = F.nll_loss(log_preds, target, reduction=self.reduction, ignore_index=self.ignore_index)

        return loss*self.eps/c + (1-self.eps) * nll


if __name__=='__main__':
    import numpy as np
    # https://blog.csdn.net/importpygame/article/details/108897682
    # https://github.com/lonePatient/label_smoothing_pytorch/blob/master/lsr.py
    pre =torch.tensor([[4.0, 5.0, 10.0], [1.0, 5.0, 4.0], [1.0, 15.0, 4.0]], dtype=torch.float)
    y = torch.tensor([2, 1, 1], dtype=torch.long)

    new_y = y * (1 - 0.1) + 0.1 / 3

    lsce = LabelSmoothCELoss(0.001)

    ce = nn.CrossEntropyLoss()

    out1 = lsce(pre, y)
    print(out1)

    out2 = ce(pre, new_y.long())
    print(out2)