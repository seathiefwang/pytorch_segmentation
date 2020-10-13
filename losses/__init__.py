from torch.nn import CrossEntropyLoss
from .dice_loss import DiceLoss, FocalDiceLoss, GeneralizedDiceLoss, CE_DiceLoss, CE_GDiceLoss
from .focal_loss import FocalLoss
from .lovasz_loss import LovaszSoftmax, CE_LovaszLoss
