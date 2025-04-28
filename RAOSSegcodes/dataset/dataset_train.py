
from torch.utils.data import DataLoader
import os
import sys
import torch
from torch.utils.data import Dataset as dataset
import nibabel as nib
import numpy as np
import random
try:
    from scipy.special import comb
except:
    from scipy.misc import comb

def to1(img):
    min_val = img.min()
    max_val = img.max()
    img = (img - min_val) / (max_val - min_val + 1e-5)  # 图像归一化
    return img

def hu_300_to1(img):
    img[np.where(img <= -300)]= -300
    img[np.where(img >= 300)] = 300
    min_val = img.min()
    max_val = img.max()
    img = (img - min_val) / (max_val - min_val + 1e-5)  # 图像归一化
    return img

import torch.nn.functional as F


class Train_Dataset(dataset):
    def __init__(self, images_names):
        self.images_list = images_names
    def __getitem__(self, index):
        ct_array = nib.load(str(self.images_list[index])).get_fdata()
        seg_array = nib.load(str(self.images_list[index]).replace("images", "labels")).get_fdata()
        # ct_array = hu_300_to1(ct_array)
        # seg_array[seg_array < 10.] = 0.
        # seg_array[seg_array > 14.] = 0.
        # seg_array[seg_array == 11.] = 0.
        # seg_array[seg_array == 12.] = 0.
        # seg_array[seg_array == 13.] = 0.
        # seg_array[seg_array == 10.] = 1.
        # seg_array[seg_array == 14.] = 1.
    
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0).unsqueeze(0)
        ct_array_resize = F.interpolate(ct_array, size=([128, 128, 160]), mode='trilinear', align_corners=False)
        ct_array_resize = ct_array_resize.squeeze(0).squeeze(0)
        ct_array_resize = ct_array_resize.numpy()

        seg_array = torch.FloatTensor(seg_array).unsqueeze(0).unsqueeze(0)
        seg_array_resize = F.interpolate(seg_array, size=([128, 128, 160]), mode="nearest")
        seg_array_resize = seg_array_resize.squeeze(0).squeeze(0)
        seg_array_resize = seg_array_resize.numpy()

        ct_array_resize = ct_array_resize.astype(np.float32)
        ct_array_resize = torch.FloatTensor(ct_array_resize).unsqueeze(0)
        seg_array_resize = torch.FloatTensor(seg_array_resize)

        return ct_array_resize, seg_array_resize

    def __len__(self):
        return len(self.images_list)
