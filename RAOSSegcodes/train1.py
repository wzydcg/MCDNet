import glob
import os
import random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from dataset.dataset_val import Val_Dataset
from dataset.dataset_train import Train_Dataset
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
from tqdm import tqdm
from models import UNet
from models import UNet_dsc
from models import Our_UNet
from utils import logger, weights_init, metrics, common, lovasz_losses
from utils import loss
import os
import numpy as np
from collections import OrderedDict
'''
训练和验证的的代码
'''
def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def val(model, val_loader, loss_func, n_labels):
    print('开始本轮验证')
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    val_hausdorff = metrics.HausdorffAverage(n_labels)

    with torch.no_grad():
        for idx, (data, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            data, target = data.float(), target.long()
            data, target = data.to(device), target.to(device)
            output = model(data)
            target = common.to_one_hot_3d(target, n_labels)
            loss = loss_func(output, target)
            val_loss.update(loss.item(), data.size(0))
            val_dice.update(output, target)
            val_hausdorff.update(output, target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg,
                           'Val_dice_label_mean': np.mean(val_dice.avg[:]),
                           'Val_hausdorff_label_mean': np.mean(val_hausdorff.avg[:])
                           })
    return val_log


def train(model, train_loader, optimizer, loss_func, n_labels):
    print("=======Epoch:{}=======lr:{}".format(epoch, optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)
    train_hausdorff = metrics.HausdorffAverage(n_labels)
    all_preds, all_label = [], []
    for idx, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        # print("data:",data)
        # print("target:",target)
        data, target = data.float(), target.long()
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        target = common.to_one_hot_3d(target, n_labels)
        loss = loss_func(output, target)
        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.item(), data.size(0))
        train_dice.update(output, target)
        train_hausdorff.update(output, target)
    train_log = OrderedDict({'Train_Loss': train_loss.avg,
                           'Train_dice_label_mean': np.mean(train_dice.avg[:]),
                             'Train_hausdorff_label_mean': np.mean(train_hausdorff.avg[:])
                             })
    return train_log


if __name__ == '__main__':

    save_path = r'/root/RAOSSeg/trainingrecords_our'   # 保存训练记录的地址
    os.makedirs(save_path, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data info
    dataset_images_train = glob.glob(r'/root/autodl-tmp/CAS2023_trainingdataset/images/*.nii.gz') # 训练集的数据地址
    random.shuffle(dataset_images_train)
    print('train:', dataset_images_train)
    dataset_images_val = glob.glob(r'/root/autodl-tmp/CAS2023_trainingdataset/Val_images/*.nii.gz')# 验证集的数据地址
    print('val:', dataset_images_val)
    train_loader = DataLoader(dataset=Train_Dataset(dataset_images_train), batch_size=2,
                              num_workers=0, shuffle=True)
    val_loader = DataLoader(dataset=Val_Dataset(dataset_images_val), batch_size=2,
                            num_workers=0, shuffle=False)
    labels_nums = 2
    epochs = 10000
    # model info
    model_name = 'our'
    if model_name == 'unet':
        model = UNet(in_channel=1, out_channel=labels_nums).to(device)
    elif model_name == 'unet_dsc':
        model = UNet_dsc(in_channel=1, out_channel=labels_nums).to(device)
    elif model_name == 'our':
        model = Our_UNet.Our_3dUNet(in_channel=1, num_classes=labels_nums).to(device)


    model.apply(weights_init.weights_init_xavier)  # 模型权重初始化，正态分布
    # dict = torch.load(r'/root/RAOSSeg/weights/best_model.pth')
    # model.load_state_dict(dict, strict=False)

    optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay = 0.0005) # 加大学习率,设置了权重衰减
    device = torch.device('cuda')
    model = model.to(device)  # multi-GPU
    # loss = lovasz_losses.lovasz_softmax
    loss = loss.HybridLoss()
    # loss = loss.TverskyLoss()

    log = logger.Train_Logger(save_path, "train_log")
    best = [0] * 20  # 初始化最优模型的epoch和performance
    best_dice = [0, 0, 0]
    max_eval_dice = 0
    best_hausdorff = [0, 100, 100]
    trigger = 0  # early stop 计数器

    for epoch in range(1, epochs):
        '''
        训练与验证
        '''
        train_log = train(model, train_loader, optimizer, loss, labels_nums)
        val_log = val(model, val_loader, loss, labels_nums)
        log.update(epoch, train_log, val_log)
        # Save checkpoint.
        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        # 保存每一轮权重
        # if epoch % 10 == 0:
        #     torch.save(state, os.path.join(save_path, f'model_{epoch}.pth'))
        '''
        保存验证上的指标信息
        '''
        if val_log['Val_dice_label_mean'] > max_eval_dice:
            max_eval_dice = val_log['Val_dice_label_mean']
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice_label_mean']
            best[2] = val_log['Val_hausdorff_label_mean']
            trigger = 0
        print('Best performance at Epoch: {} | Val_dice_label_mean: {} | Val_hausdorff_label_mean: {} '
              .format(best[0], best[1], best[2]))
        torch.cuda.empty_cache()