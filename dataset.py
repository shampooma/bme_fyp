import os
import cv2
import math
import torch
import numpy as np

from skimage.transform import rotate

from torch.utils import data

class Dataset(data.Dataset):
    def __init__(
        self,
        split,
        do_transform,
    ):
        self.split = split
        self.do_transform = do_transform

        self.files = os.listdir(f"./data/{split}/images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # Read image and gt
        img_name = self.files[index]
        img_path = f"./data/{self.split}/images/{img_name}"
        gt_path = f"./data/{self.split}/gts/{img_name}"

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).astype(np.float)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE).astype(np.float)

        # Change gt
        gt[gt==11] = 3
        gt[gt==9] = 2

        # Transform
        if self.do_transform:
            img, gt= self.transform(img, gt)

        # Preprocessing
        img, gt = self.preprocessing(img, gt)

        # Normalize
        img = (img - img.mean()) / img.std()

        # HW -> CHW
        img = np.expand_dims(img, axis=0)

        # To tensor
        img = torch.from_numpy(img.copy()).float()
        gt = torch.from_numpy(gt.copy()).long()

        return img, gt

    def preprocessing(self, img, gt):
        # 1. Convert from catesian to polar coordinate
        # 2. Cut the unused parts
        # 3. Rotate image to be good looking
        img = cv2.warpPolar(
            img,
            (math.ceil(1250/2), math.ceil(1250*np.pi/16)*16),
            (1250/2, 1250/2),
            1250/2,
            cv2.INTER_CUBIC + cv2.WARP_POLAR_LINEAR
        )
        img = img[:,-352::]
        img = img.T
        img = img[::-1,:]

        gt = cv2.warpPolar(
            gt,
            (math.ceil(1250/2), math.ceil(1250*np.pi/16)*16),
            (1250/2, 1250/2),
            1250/2,
            cv2.INTER_NEAREST + cv2.WARP_POLAR_LINEAR
        )
        gt = gt[:,-352::]
        gt = gt.T
        gt = gt[::-1,:]

        return img, gt

    def transform(self, img, gt):
        img_size = np.array([img.shape[0], img.shape[1]])

        # Pad
        crop_pad_ratio = np.random.rand(1)[0]*0.05
        crop_pad_value = np.round(img_size*crop_pad_ratio).astype(np.int)

        img = np.pad(img, [
            [crop_pad_value[0]//2, crop_pad_value[0]-crop_pad_value[0]//2],
            [crop_pad_value[1]//2, crop_pad_value[1]-crop_pad_value[1]//2]
        ], constant_values=254)
        gt = np.pad(gt, [
            [crop_pad_value[0]//2, crop_pad_value[0]-crop_pad_value[0]//2],
            [crop_pad_value[1]//2, crop_pad_value[1]-crop_pad_value[1]//2]
        ], constant_values=0)

        # Horizontal flip
        flip = np.random.rand(1)[0]
        if flip >= 0.5:
            img = cv2.flip(img, 1)
            gt = cv2.flip(gt, 1)

        # Rotate
        angle = np.random.rand(1)[0]*360
        img = rotate(
            image=img, 
            angle=angle, 
            order=3, 
            cval=254
        )
        gt = rotate(
            image=gt, 
            angle=angle, 
            order=0, 
            cval=0
        )

        return img, gt

if __name__ == "__main__":
    pass


