from base import BaseDataSet, BaseDataLoader
from utils import palette
import numpy as np
import os
import scipy
import torch
from PIL import Image
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class RSIDataset(BaseDataSet):

    def __init__(self, num_classes=14, **kwargs):
        self.num_classes = num_classes
        self.palette = palette.get_voc_palette(self.num_classes)
        super(RSIDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, 'train')
        self.image_dir = os.path.join(self.root, 'image')
        self.label_dir = os.path.join(self.root, 'label')

        file_list = os.path.join(self.root, self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]

    def _get_label(self, label):
        # 类别对应
        if self.num_classes == 8:
            matches = [800, 100, 200, 300, 400, 500, 600, 700]
        elif self.num_classes == 17:
            matches = [17, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        elif self.num_classes == 15:
            matches = [17, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        elif self.num_classes == 14:
            matches = [17, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        h, w = label.shape
        seg_labels = np.zeros((w, h), dtype=np.uint8)

        for i in range(self.num_classes):
            seg_labels[label == matches[i]] = i
        
        seg_labels[label == 0] = 0
        seg_labels[label == 255] = 255
        if self.num_classes == 14:
            seg_labels[label == 4] = 255

        return seg_labels

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.tif')
        label_path = os.path.join(self.label_dir, image_id + '.png')
        image = np.asarray(Image.open(image_path))
        label = np.asarray(Image.open(label_path), dtype=np.int32)
        label = self._get_label(label)
        # image_id = self.files[index].split("/")[-1].split(".")[0]
        return image, label, image_id

    def get_weights(self):
        save_path = os.path.join(self.root, 'weights.npy')
        if os.path.exists(save_path):
            weights = np.load(save_path)
            return weights

        # class_weight = {1:0.065, 2:0.07, 3:0.07, 4:0.01, 7:0.05, 8:0.01, 9:0.07, 10:0.04, 11:0.07, 12:0.04, 13:0.07, 14:0.065, 15:0.045, 16:0.025, 17:0.07}
        class_weight = {1:0.065, 2:0.07, 3:0.07, 4:0, 7:0.05, 8:0.016, 9:0.07, 10:0.04, 11:0.07, 12:0.04, 13:0.07, 14:0.065, 15:0.045, 16:0.03, 17:0.07}

        # class_sum = {2: 1543114305, 7: 1311472725, 11: 2476695927, 13: 3064837353, 17: 473091372, 3: 2371671666, 9: 4478857437, 16: 221682972, 10: 247614603, 14: 1439428722, 8: 52628364, 12: 470929422, 1: 541517850, 15: 966983106, 4: 274176}
        class_num = {2: 73904, 7: 28175, 11: 75679, 13: 73239, 17: 65397, 3: 64540, 9: 47947, 16: 7874, 10: 12158, 14: 44790, 8: 3101, 12: 15291, 1: 32271, 15: 20300, 4: 14}
        # class_num = {2: 73904, 7: 28175, 11: 75679, 13: 73239, 17: 65397, 3: 64540, 9: 47947, 16: 7874, 10: 12158, 14: 44790, 8: 3101, 12: 15291, 1: 32271, 15: 20300, 4: 1261}
        # class_num = {2: 73904, 7: 28175, 11: 75679, 13: 73239, 17: 65397, 3: 64540, 9: 47947, 16: 7874, 10: 12158, 14: 44790, 8: 3101, 12: 15291, 1: 32271, 15: 20300, 4: 2152}
        # class_num = {2: 73904, 7: 28175, 11: 75679, 13: 73239, 17: 65397, 3: 64540, 9: 47947, 16: 7874, 10: 12158, 14: 44790, 8: 3101, 12: 15291, 1: 32271, 15: 20300, 4: 3750} #1598


        class_weight = {1:0.065, 2:0.07, 3:0.07, 4:0, 7:0.06, 8:0.03, 9:0.07, 10:0.04, 11:0.07, 12:0.05, 13:0.07, 14:0.07, 15:0.05, 16:0.03, 17:0.07}
        class_num = {1: 42599, 2: 85115, 3: 78318, 9: 48372, 10: 12171, 11: 81375, 13: 78563, 14: 53880, 17: 70632, 15: 24629, 16: 8359, 12: 25642, 7: 46717, 8: 5452, 4: 14, 255: 29379}

        weights = []
        for image_id in self.files:
            label_path = os.path.join(self.label_dir, image_id + '.png')
            label = cv2.imread(label_path)
            key = np.unique(label)

            wei = 0
            for k in key:
                if k == 0 or k == 255: continue
                # mask = label == k
                # label_n = np.sum(mask)
                w = class_weight[k] * (10000 / class_num[k])
                wei = w if w > wei else wei
            weights.append(wei)

        weights = np.asarray(weights)
        np.save(save_path, weights)
        return weights

class RSIFastDataset(BaseDataSet):

    def __init__(self, **kwargs):
        self.num_classes = 8
        self.palette = palette.get_voc_palette(self.num_classes)
        super(RSIFastDataset, self).__init__(**kwargs)

    def _set_files(self):
        self.root = os.path.join(self.root, 'train')
        self.image_dir = os.path.join(self.root, 'image')
        self.label_dir = os.path.join(self.root, 'label')

        file_list = os.path.join(self.root, self.split + ".txt")
        self.files = [line.rstrip() for line in tuple(open(file_list, "r"))]

        self.images = []
        self.labels = []
        for image_id in self.files:
            # image_path = os.path.join(self.image_dir, image_id + '.tif')
            label_path = os.path.join(self.label_dir, image_id + '.png')
            # image = np.asarray(Image.open(image_path), dtype=np.float32)
            label = np.asarray(Image.open(label_path), dtype=np.int32)
            label = self._get_label(label)
            # self.images.append(image)
            self.labels.append(label)


    def _get_label(self, label):
        # 类别对应
        if self.num_classes == 8:
            matches = [100, 200, 300, 400, 500, 600, 700, 800]
        elif self.num_classes == 17:
            matches = [100, 200, 300, 400, 500, 600, 700, 800]

        h, w = label.shape
        seg_labels = np.zeros((w, h))

        for i in range(self.num_classes):
            seg_labels[label == matches[i]] = i
        
        return seg_labels

    def _load_data(self, index):
        image_id = self.files[index]
        image_path = os.path.join(self.image_dir, image_id + '.tif')
        image = np.asarray(Image.open(image_path), dtype=np.float32)
        label = self.labels[index]
        return image, label, image_id

class RSI(BaseDataLoader):
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, num_classes=14, num_workers=1, val=False,
                    shuffle=False, val_split=None, return_id=False, **kwargs):
        
        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

        kwargs = {
            'num_classes': num_classes,
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'crop_size': crop_size,
            'base_size': base_size,
            'return_id': return_id,
            'val': val
        }
    
        # if split in ["train_fast", "trainval_fast", "val_fast", "test_fast"]:
        #     self.dataset = RSIFastDataset(**kwargs)
        # elif split in ["train", "train_all", "train_aug", "val", "test"]:
        #     self.dataset = RSIDataset(**kwargs)
        # else: raise ValueError(f"Invalid split name {split}")

        self.dataset = RSIDataset(**kwargs)

        if 'train' in split:
            weights = self.dataset.get_weights()
            # weights = None
        else:
            weights = None
        super(RSI, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split, weights)

