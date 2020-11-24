import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scipy import ndimage
import albumentations as albu

class BaseDataSet(Dataset):
    def __init__(self, root, split, mean, std, base_size=None, augment=True, val=False,
                crop_size=321, scale=True, flip=True, rotate=False, blur=False, return_id=False):
        self.root = root
        self.split = split
        self.mean = mean
        self.std = std
        self.augment = augment
        self.crop_size = crop_size
        if self.augment:
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur
        self.val = val
        self.files = []
        self._set_files()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)
        self.return_id = return_id

        if not self.val:
            self.affine_augmenter = albu.Compose([
                            albu.Flip(p=0.3),
                            albu.RandomRotate90(p=0.3),
                            albu.Rotate(limit=45, value=0, mask_value=0, border_mode=cv2.BORDER_CONSTANT, p=0.005)
                            ])

            self.image_augmenter = albu.Compose([
                            albu.OneOf([
                                albu.GaussNoise(var_limit=(0, 15)),
                                # albu.IAAAdditiveGaussianNoise(),
                                ], p=0.01),
                            albu.OneOf([
                                # albu.MotionBlur(blur_limit=3),
                                albu.GaussianBlur(3),
                                # albu.Blur(blur_limit=3),
                                ], p=0.01),
                            albu.RandomBrightnessContrast(0.15,0.15, p=0.05),
                            albu.OneOf([
                                albu.CLAHE(clip_limit=2),# 
                                albu.IAASharpen(),# 锐化
                                # albu.IAAEmboss(),# 浮雕
                                albu.RandomGamma(),
                                ], p=0.001),
                            albu.HueSaturationValue(hue_shift_limit=0, sat_shift_limit=10, val_shift_limit=20,p=0.02),
                            albu.JpegCompression(90, 100, p=0.001)
                            ])

            self.resizer = albu.Compose([
                            albu.Resize(base_size, base_size, p=1),
                            albu.RandomScale(scale_limit=(-0.2, 0.25), p=0.05),
                            albu.PadIfNeeded(min_height=crop_size, min_width=crop_size,
                                            value=0, mask_value=0, border_mode=cv2.BORDER_CONSTANT, p=1),
                            albu.RandomCrop(height=crop_size, width=crop_size, p=1)
                            ])
        else:
            self.affine_augmenter = None
            self.image_augmenter = None
            self.resizer = albu.Compose([
                            albu.Resize(crop_size, crop_size, p=1)
                            ])

        cv2.setNumThreads(0)

    def _set_files(self):
        raise NotImplementedError
    
    def _load_data(self, index):
        raise NotImplementedError

    def _val_augmentation(self, image, label):
        if self.crop_size:
            h, w = label.shape
            # Scale the smaller side to crop size
            if h < w:
                h, w = (self.crop_size, int(self.crop_size * w / h))
            else:
                h, w = (int(self.crop_size * h / w), self.crop_size)

            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = Image.fromarray(label).resize((w, h), resample=Image.NEAREST)
            label = np.asarray(label, dtype=np.int32)

            # Center Crop
            h, w = label.shape
            start_h = (h - self.crop_size )// 2
            start_w = (w - self.crop_size )// 2
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]
        return image, label

    def _augmentation(self, image, label):
        h, w, _ = image.shape
        # Scaling, we set the bigger to base size, and the smaller 
        # one is rescaled to maintain the same ratio, if we don't have any obj in the image, re-do the processing
        if self.base_size:
            if self.scale:
                longside = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
            else:
                longside = self.base_size
            h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
    
        h, w, _ = image.shape
        # Rotate the image with an angle between -10 and 10
        if self.rotate:
            angle = random.randint(-10, 10)
            center = (w / 2, h / 2)
            rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_LINEAR)#, borderMode=cv2.BORDER_REFLECT)
            label = cv2.warpAffine(label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)#,  borderMode=cv2.BORDER_REFLECT)

        # Padding to return the correct crop size
        if self.crop_size:
            pad_h = max(self.crop_size - h, 0)
            pad_w = max(self.crop_size - w, 0)
            pad_kwargs = {
                "top": 0,
                "bottom": pad_h,
                "left": 0,
                "right": pad_w,
                "borderType": cv2.BORDER_CONSTANT,}
            if pad_h > 0 or pad_w > 0:
                image = cv2.copyMakeBorder(image, value=0, **pad_kwargs)
                label = cv2.copyMakeBorder(label, value=0, **pad_kwargs)
            
            # Cropping 
            h, w, _ = image.shape
            start_h = random.randint(0, h - self.crop_size)
            start_w = random.randint(0, w - self.crop_size)
            end_h = start_h + self.crop_size
            end_w = start_w + self.crop_size
            image = image[start_h:end_h, start_w:end_w]
            label = label[start_h:end_h, start_w:end_w]

        # Random H flip
        if self.flip:
            if random.random() > 0.5:
                image = np.fliplr(image).copy()
                label = np.fliplr(label).copy()

        # Gaussian Blud (sigma between 0 and 1.5)
        if self.blur:
            sigma = random.random()
            ksize = int(3.3 * sigma)
            ksize = ksize + 1 if ksize % 2 == 0 else ksize
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        return image, label
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image, label, image_id = self._load_data(index)
        # if self.val:
        #     image, label = self._val_augmentation(image, label)
        # elif self.augment:
        #     image, label = self._augmentation(image, label)

        if not self.val:
            augmented = self.affine_augmenter(image=image, mask=label)
            image, label = augmented['image'], augmented['mask']
            
            augmented = self.image_augmenter(image=image)
            image = augmented['image']

        augmented = self.resizer(image=image, mask=label)
        image, label = augmented['image'], augmented['mask']


        label = torch.from_numpy(np.array(label, dtype=np.int32)).long()
        image = Image.fromarray(np.uint8(image))
        if self.return_id:
            return  self.normalize(self.to_tensor(image)), label, image_id
        return self.normalize(self.to_tensor(image)), label

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str

