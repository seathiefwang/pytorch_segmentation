from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, einsum
from .focal_loss import FocalLoss
from .smooth_loss import LabelSmoothCELoss

def make_one_hot(labels, classes):
    one_hot = torch.FloatTensor(labels.size()[0], classes, labels.size()[2], labels.size()[3]).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    return target

class DiceLoss(nn.Module):
    def __init__(self, smooth=0, eps=1e-10, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, preds, labels):
        classes = preds.size()[1]
        preds = F.softmax(preds, dim=1)

        if self.ignore_index is not None:
            labels.requires_grad = False
            mask = labels != self.ignore_index
            labels[labels == self.ignore_index] = labels.min()
            # labels = labels[mask]

        labels = make_one_hot(labels.unsqueeze(dim=1), classes=classes)
        if self.ignore_index is not None:
            labels = labels.permute(0, 2, 3, 1)
            preds = preds.permute(0, 2, 3, 1)
            labels = labels[mask]
            preds = preds[mask]

        preds_flat = preds.contiguous().view(-1)
        labels_flat = labels.contiguous().view(-1)
        intersection = (preds_flat * labels_flat).sum()
        loss = 1 - ((2. * intersection + self.smooth) /
                    (preds_flat.sum() + labels_flat.sum() + self.smooth + self.eps))
        return loss


class GeneralizedDiceLoss(nn.Module):
    def __init__(self, smooth=0, eps=1e-10, ignore_index=255):
        super().__init__()
        self.smooth = smooth
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, preds, labels):
        classes = preds.size()[1]
        if self.ignore_index not in range(labels.min(), labels.max()):
            if (labels == self.ignore_index).sum() > 0:
                labels[labels == self.ignore_index] = labels.min()

        labels = make_one_hot(labels.unsqueeze(dim=1), classes=classes)

        pc = preds.type(torch.float32)
        tc = labels.type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

        loss = divided.mean()

        return loss



class FocalDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.5, focal_weight=0.5, ignore_index=255):
        super(FocalDiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.dice = DiceLoss()
        self.focal = FocalLoss(ignore_index=ignore_index)
    
    def forward(self, output, target):
        focal_loss = self.focal(output, target)
        dice_loss = self.dice(output, target)
        # print("focal dice :", focal_loss.item(), dice_loss.item())
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss

class CE_DiceLoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5, ignore_index=255):
        super(CE_DiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        # print("ce dice :", CE_loss.item(), dice_loss.item())
        return self.ce_weight * CE_loss + self.dice_weight * dice_loss

class CE_GDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5, ignore_index=255):
        super(CE_GDiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = GeneralizedDiceLoss()
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        # print("ce dice :", CE_loss.item(), dice_loss.item())
        return self.ce_weight * CE_loss + self.dice_weight * dice_loss

class ExDiceLoss(nn.Module):
    def __init__(self, smooth=0, gamma=1, eps=1e-10, ignore_index=255):
        super().__init__()
        self.gamma = gamma
        self.dice = DiceLoss(smooth=0, eps=1e-10, ignore_index=255)

    def forward(self, output, target):
        dice_score = self.dice(output, target)
        logarithmic_exp = (-1 * torch.log(dice_score)) ** self.gamma
        per_class_ds = logarithmic_exp.mean()
        return per_class_ds


class SmoothCE_DiceLoss(nn.Module):
    def __init__(self, dice_weight=0.5, ce_weight=0.5, ignore_index=255):
        super(SmoothCE_DiceLoss, self).__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.dice = DiceLoss(ignore_index=255)
        self.cross_entropy = LabelSmoothCELoss(ignore_index=ignore_index)
    
    def forward(self, output, target):
        CE_loss = self.cross_entropy(output, target)
        dice_loss = self.dice(output, target)
        # print("ce dice :", CE_loss.item(), dice_loss.item())
        return self.ce_weight * CE_loss + self.dice_weight * dice_loss


