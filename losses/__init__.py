import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from .dice_loss import DiceLoss, FocalDiceLoss, GeneralizedDiceLoss, \
                    CE_DiceLoss, CE_GDiceLoss, SmoothCE_DiceLoss
from .focal_loss import FocalLoss
from .lovasz_loss import LovaszSoftmax, CE_LovaszLoss
from .smooth_loss import LabelSmoothCELoss


class Criterion(nn.Module):
    def __init__(self, loss_type='MSE', **kwargs):
        super().__init__()
        if loss_type == 'MSE':
            self.criterion = nn.CrossEntropyLoss(**kwargs)
        elif loss_type == 'CE_DiceLoss':
            self.criterion = CE_DiceLoss(**kwargs)
        elif loss_type == 'LabelSmoothCELoss':
            self.criterion = LabelSmoothCELoss(**kwargs)
        elif loss_type == 'SmoothCE_DiceLoss':
            self.criterion = SmoothCE_DiceLoss(**kwargs)
        else:
            raise NotImplementedError

    def forward(self, preds, labels):
        number = len(preds) - 1
        loss = self.criterion(preds[0],labels)
        for i in range(1, len(preds)):
            labels = labels.unsqueeze(1).float()
            labels = F.interpolate(labels, scale_factor=0.5, mode='nearest')
            labels = labels.squeeze().long()
            loss += 1 / number * self.criterion(preds[i],labels)

        return loss
