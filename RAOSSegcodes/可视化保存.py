import glob

import nibabel as nib
import numpy as np
import cv2
import matplotlib.pyplot as plt

names=  glob.glob(r'H:\RAOSSeg\RAOS-Real\CancerImages(Set1)\imagesTr\1.2.840.113619.2.416.10235540276395849575842699527818556338.nii.gz')

def hu_300_to1(img):
    img[np.where(img <= -300)] = -300
    img[np.where(img >= 300)] = 300
    min_val = img.min()
    max_val = img.max()
    img = (img - min_val) / (max_val - min_val + 1e-5)  # 图像归一化
    return img * 255


for name in names:
    print(name)
    ct_array = nib.load(name).get_fdata()
    seg_array = nib.load(name.replace('images', 'labels')).get_fdata()

    ct_array = hu_300_to1(ct_array)
    seg_array[seg_array < 10.] = 0.
    seg_array[seg_array > 14.] = 0.
    seg_array[seg_array == 11.] = 0.
    seg_array[seg_array == 12.] = 0.
    seg_array[seg_array == 13.] = 0.
    seg_array[seg_array == 10.] = 125.
    seg_array[seg_array == 14.] = 250.

    z = ct_array.shape[2]
    for i in range(z):
        slice = ct_array[:, :, i]
        seg_slice = seg_array[:, :, i]
        cv2.imwrite(r'H:\RAOSSeg\sliceviausl' + '\\' + str(i) + '.jpg', slice)
        cv2.imwrite(r'H:\RAOSSeg\sliceviausl' + '\\' + str(i) + '_seg.jpg', seg_slice)

