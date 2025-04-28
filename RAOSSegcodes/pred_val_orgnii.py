# -*- coding: utf-8 -*-
import glob
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from dataset.dataset_val import Val_Dataset_name_orgshape_nii as Val_Dataset
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
from utils import metrics, common, lovasz_losses
import os
import numpy as np
from collections import OrderedDict
from models import UNet
from models import UNet_dsc
from models import Our_UNet
import nibabel as nib


def to255(img):
    min_val = img.min()
    max_val = img.max()
    img = (img - min_val) / (max_val - min_val + 1e-5)  # 图像归一化
    img = img * 255  # *255
    return img

def interpolater2_mask(img, org_full_size):
    img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
    img = torch.nn.functional.interpolate(
        img, size=org_full_size, mode='nearest').numpy().squeeze(0).squeeze(0)
    return img


def val(model, val_loader, loss_func, n_labels):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    val_hausdorff = metrics.HausdorffAverage(n_labels)
    i = 0
    with torch.no_grad():
        for idx, (data, target, org_full_size, img_name
                  ) in tqdm(enumerate(val_loader), total=len(val_loader)):
            i += 1
            data, target = data.float(), target.long()
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = loss_func(output, target)
            target = common.to_one_hot_3d(target, n_labels)
            val_loss.update(loss.item(), data.size(0))
            val_dice.update(output, target)
            print(val_dice.value[1])
            print(img_name[0])
            print('*'*60)
            val_hausdorff.update(output, target)
            '''
            下面进行展示
            '''

            pred_mask = torch.argmax(output, 1).cpu().numpy()
            this_batch_pred_mask = pred_mask[0, :, :, :].astype(np.uint8)
            print(this_batch_pred_mask.max())

            this_batch_pred_mask = interpolater2_mask(this_batch_pred_mask, org_full_size)

            img_name = img_name[0]
            this_name = img_name.split('/')[-1]

            img = nib.load(img_name)
            img_affine = img.affine

            nib.Nifti1Image(this_batch_pred_mask, img_affine).to_filename(pred_label_visual_path + '\\' + this_name.replace('.nii', '_pred.nii'))

    val_log = OrderedDict({'Val_Loss': val_loss.avg,
                           'Val_dice_label_mean': np.mean(val_dice.avg[:]),
                           'Val_hausdorff_label_mean': np.mean(val_hausdorff.avg[:])
                           })

    return val_log


if __name__ == '__main__':

    weights_save_path = r'/root/RAOSSeg/trainingrecords_our/latest_model.pth'   # 训练产生的模型权重
    pred_label_visual_path = r'/root/RAOSSeg/trainingrecords_unet/visual_pred_test'
    os.makedirs(pred_label_visual_path, exist_ok=True)
    device = torch.device('cuda')
    dataset_images = glob.glob(r'/root/RAOSSeg/CAS2023_trainingdataset/Val_images/*.nii.gz')# 验证集的数据地址 # 验证集图像
    val_loader = DataLoader(dataset=Val_Dataset(dataset_images), batch_size=1,
                            num_workers=0, shuffle=False)
    labels_nums = 2
    model = Our_UNet.Our_3dUNet(in_channel=1, num_classes=labels_nums).to(device)
    # model = UNet(in_channel=1, out_channel=labels_nums).to(device)

    ckpt = torch.load(weights_save_path)
    model.load_state_dict(ckpt['net'], strict=True)
    device = torch.device('cuda')
    model = model.to(device)
    loss = lovasz_losses.lovasz_softmax
    val_log = val(model, val_loader, loss, labels_nums)
    print("val_log:", val_log)
