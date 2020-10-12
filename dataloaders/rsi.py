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

    def __init__(self, **kwargs):
        self.num_classes = 8
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
            matches = [100, 200, 300, 400, 500, 600, 700, 800]

        h, w = label.shape
        seg_labels = np.zeros((w, h), dtype=np.uint8)

        for i in range(self.num_classes):
            seg_labels[label == matches[i]] = i
            
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
    def __init__(self, data_dir, batch_size, split, crop_size=None, base_size=None, scale=True, num_workers=1, val=False,
                    shuffle=False, flip=False, rotate=False, blur= False, augment=False, val_split= None, return_id=False):
        
        self.MEAN = [0.45734706, 0.43338275, 0.40058118]
        self.STD = [0.23965294, 0.23532275, 0.2398498]

        kwargs = {
            'root': data_dir,
            'split': split,
            'mean': self.MEAN,
            'std': self.STD,
            'augment': augment,
            'crop_size': crop_size,
            'base_size': base_size,
            'scale': scale,
            'flip': flip,
            'blur': blur,
            'rotate': rotate,
            'return_id': return_id,
            'val': val
        }
    
        if split in ["train_fast", "trainval_fast", "val_fast", "test_fast"]:
            self.dataset = RSIFastDataset(**kwargs)
        elif split in ["train", "train_all", "val", "test"]:
            self.dataset = RSIDataset(**kwargs)
        else: raise ValueError(f"Invalid split name {split}")
        super(RSI, self).__init__(self.dataset, batch_size, shuffle, num_workers, val_split)

