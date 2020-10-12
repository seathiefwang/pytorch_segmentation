from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from focal_loss import FocalLoss

def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

class DiceLoss(nn.Module):
    def __init__(self, smooth=0, eps=1e-7, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, preds, labels):        
        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            labels = labels[mask]
            preds = preds[mask]
        
        target = make_one_hot(labels.unsqueeze(dim=1), classes=preds.size()[1])
        preds = F.softmax(preds, dim=1)
        preds_flat = preds.contiguous().view(-1)
        target_flat = target.contiguous().view(-1)
        intersection = (preds_flat * target_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (preds_flat.sum() + target_flat.sum() + self.smooth + self.eps))
        return loss


class FocalDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.2, focal_weight=0.8, ignore_index=255):
        super(FocalDiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = DiceLoss()
        self.focal = FocalLoss(ignore_index=ignore_index)
    
    def forward(self, output, target):
        focal_loss = self.focal(output, target)
        dice_loss = self.dice(output, target)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


