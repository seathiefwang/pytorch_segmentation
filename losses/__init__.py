import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from .dice_loss import DiceLoss, FocalDiceLoss, GeneralizedDiceLoss, CE_DiceLoss, CE_GDiceLoss
from .focal_loss import FocalLoss
from .lovasz_loss import LovaszSoftmax, CE_LovaszLoss


class Criterion(nn.Module):
    def __init__(self, loss_type='MSE', **kwargs):
        super().__init__()
        if loss_type == 'MSE':
            # self.criterion = nn.CrossEntropyLoss(**kwargs)
            self.criterion = nn.MSELoss(**kwargs)
        elif loss_type == 'L1':
            self.criterion = nn.L1Loss()
        elif loss_type == 'SMOOTHL1':
            self.criterion = nn.SmoothL1Loss()
        elif loss_type == 'WING':
            self.criterion = Wingloss()
        else:
            raise NotImplementedError

        self.y_criterion = nn.CrossEntropyLoss()
        self.p_criterion = nn.CrossEntropyLoss()
        self.r_criterion = nn.CrossEntropyLoss()

    # def forward(self, preds, labels):
    #     return self.criterion(preds,labels)
    def forward(self, preds, labels):
        angles_loss = self.criterion(preds[0],labels[0])

        yaw_loss = self.y_criterion(preds[1], labels[1])
        pitch_loss = self.p_criterion(preds[2], labels[2])
        roll_loss = self.r_criterion(preds[3], labels[3])

        # quat loss: w**2 + x**2 + y**2 + z**2 = 1
        # quat_loss = torch.sum(torch.abs(1 - torch.sum(torch.pow(preds[0], 2), 1)), 1)

        return angles_loss + 0.1*(yaw_loss+pitch_loss+roll_loss)
