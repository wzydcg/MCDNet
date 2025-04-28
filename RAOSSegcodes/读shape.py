import glob

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

names=  glob.glob(r'H:\RAOSSeg\RAOS-Real\CancerImages(Set1)\labelsTr\*.nii.gz')

for name in names:
    print(name)
    img = nib.load(name).get_fdata()
    seg = nib.load(name.replace('images', 'labels')).get_fdata()


    # print(img.shape)
    print(img.shape)
    print(np.unique(seg))
    print('*'*20)
