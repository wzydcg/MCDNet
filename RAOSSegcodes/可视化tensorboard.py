# -*- coding: utf-8 -*-
import os
import pandas as pd
from matplotlib import pyplot as plt

'''
绘制loss图
'''


def loss_visualize(epoch_loss_unet, value_loss_unet, epoch_loss_unet_dsc, value_loss_unet_dsc):
    plt.style.use('tableau-colorblind10')

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.title("Validation Dice Curve")

    plt.plot(epoch_loss_unet, value_loss_unet, label='3DUnet', color='c', linestyle='-')
    plt.plot(epoch_loss_unet_dsc, value_loss_unet_dsc, label='3DUnet DSC', color='g',
             linestyle='-')

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Dice')
    plt.grid()
    plt.savefig(os.path.join(res_dir, r'dice.eps'), dpi=350, format='eps')
    plt.savefig(os.path.join(res_dir, r'dice.png'), dpi=350, format='png')
    plt.savefig(os.path.join(res_dir, r'dice.svg'), dpi=350, format='svg')

    plt.show()


def read_value(train_df):
    epoch = train_df['Step']
    value = train_df['Value']
    return epoch, value


if __name__ == "__main__":
    root_dir = os.getcwd()
    file_dir = os.path.join(root_dir, r'H:\RAOSSeg')
    res_dir = os.path.join(root_dir, r'H:\RAOSSeg')
    loss_unet = pd.read_csv(os.path.join(file_dir, 'trainingrecords.csv'))
    loss_unet_dsc = pd.read_csv(os.path.join(file_dir, 'trainingrecords_dsc.csv'))

    epoch_loss_unet_dsc, value_loss_unet_dsc = read_value(loss_unet_dsc)
    epoch_loss_unet, value_loss_unet = read_value(loss_unet)

    loss_visualize(epoch_loss_unet, value_loss_unet, epoch_loss_unet_dsc, value_loss_unet_dsc)
