#!/usr/bin/python3
# -*- coding: utf-8 -*
import torch
from torch import nn
from models.S3_DSConv import DCN_Conv
from models.acmix import ACmix
from models.aspp import ASPPConv
from models.deform.template_conv import DeformConvPack_d

import torch.nn.functional as F

class SingleConv3DBlock_dsc(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, morph):
        super().__init__()
        self.block = DCN_Conv(in_planes, out_planes, kernel_size, 1.0, morph, True, 'cuda')

    def forward(self, x):
        y = self.block(x)

        return y

class Our_3dUNet(nn.Module):

    def __init__(self, num_classes, in_channel):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder1 = nn.Sequential(DeformConvPack_d(in_channels=in_channel,out_channels=32,kernel_size=3,stride=1, padding=1, dimension='HW').cuda(),nn.BatchNorm3d(32))
        self.encoder2 = nn.Sequential(DeformConvPack_d(in_channels=32,out_channels=64,kernel_size=3,stride=1, padding=1, dimension='HW').cuda(),nn.BatchNorm3d(64))
        self.encoder3 = nn.Sequential(DeformConvPack_d(in_channels=64,out_channels=128,kernel_size=3,stride=1, padding=1, dimension='HW').cuda(),nn.BatchNorm3d(128))
        self.encoder4 = nn.Sequential(DeformConvPack_d(in_channels=128,out_channels=256,kernel_size=3,stride=1, padding=1, dimension='HW').cuda(),nn.BatchNorm3d(256))  
        self.encoder5 = nn.Sequential(DeformConvPack_d(in_channels=256,out_channels=512,kernel_size=3,stride=1, padding=1, dimension='HW').cuda(),nn.BatchNorm3d(512))
        
        self.encoder2_y = nn.Sequential(nn.Conv3d(32, 64, 3,stride = 1,padding = 1),nn.BatchNorm3d(64))
        self.encoder3_y = nn.Sequential(nn.Conv3d(64, 128, 3,stride = 1,padding = 1),nn.BatchNorm3d(128))
        self.encoder4_y = nn.Sequential(nn.Conv3d(128, 256, 3,stride = 1,padding = 1),nn.BatchNorm3d(256))
        self.encoder5_y = nn.Sequential(nn.Conv3d(256, 512, 3,stride = 1,padding = 1),nn.BatchNorm3d(512))
        
        # self.conv4 = nn.Sequential(ASPPConv(256, 256, [6, 12, 18]), nn.BatchNorm3d(256)) #bottle_neck处使用aspp卷积
        
        self.decoder1 = nn.Sequential(nn.Conv3d(512 * 2, 512, 3, stride=1, padding=1),
                                      nn.BatchNorm3d(512))
        self.decoder2 = nn.Sequential(nn.Conv3d(512 + 256 * 2, 256, 3, stride=1, padding=1),
                                      nn.BatchNorm3d(256))    
        self.decoder3 = nn.Sequential(nn.Conv3d(256 + 128 * 2, 128, 3, stride=1, padding=1),
                                      nn.BatchNorm3d(128))
        self.decoder4=  nn.Sequential(nn.Conv3d(128 + 64 * 2, 64, 3, stride=1, padding=1),
                                      nn.BatchNorm3d(64))
        self.decoder5 = nn.Sequential(nn.Conv3d(64 + 32 * 2, 32, 3, stride=1, padding=1),
                                      nn.BatchNorm3d(32))
        
        self.map4 = nn.Sequential(
            nn.Conv3d(32, num_classes, 1, 1),
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=True),
            nn.Softmax(dim=1)
        )
        self.acmix1 = nn.Sequential(ACmix(in_planes = 32, out_planes = 64),nn.BatchNorm3d(64))
        self.acmix2 = nn.Sequential(ACmix(in_planes = 64, out_planes = 128),nn.BatchNorm3d(128))
        self.acmix3 = nn.Sequential(ACmix(in_planes = 128, out_planes = 256),nn.BatchNorm3d(256))
        self.acmix4 = nn.Sequential(ACmix(in_planes = 256, out_planes = 512),nn.BatchNorm3d(512))

    def forward(self, input):
        # 分两个分支进行下采样
        # 采用改进的模板卷积下采样分支 
        x0_0 = F.relu(F.max_pool3d(self.encoder1(input), 2, 2))
        x1_0 = F.relu(F.max_pool3d(self.encoder2(x0_0), 2, 2))
        x2_0 = F.relu(F.max_pool3d(self.encoder3(x1_0), 2, 2))
        x3_0 = F.relu(F.max_pool3d(self.encoder4(x2_0), 2, 2))
        x4_0 = F.relu(F.max_pool3d(self.encoder5(x3_0), 2, 2))
      
        #采用3d-acmix进行下采样的分支
        y0_0 = F.relu(F.max_pool3d(self.encoder1(input), 2, 2))
        y1_0 = F.relu(F.max_pool3d(self.encoder2_y(y0_0), 2, 2))
        y2_0 = F.relu(F.max_pool3d(self.encoder3_y(y1_0), 2, 2))
        y3_0 = F.relu(F.max_pool3d(self.encoder4_y(y2_0), 2, 2))
        y4_0 = F.relu(F.max_pool3d(self.acmix4(y3_0), 2, 2))
        
        u0 = F.relu(F.interpolate(self.decoder1(torch.cat([x4_0, y4_0], dim = 1)), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True))        
        u1 = F.relu(F.interpolate(self.decoder2(torch.cat([u0, x3_0, y3_0], dim = 1)), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)) 
        u2 = F.relu(F.interpolate(self.decoder3(torch.cat([u1, x2_0, y2_0], dim = 1)), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)) 
        u3 = F.relu(F.interpolate(self.decoder4(torch.cat([u2, x1_0, y1_0], dim = 1)), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)) 
        u4 = F.relu(F.interpolate(self.decoder5(torch.cat([u3, x0_0, y0_0], dim = 1)), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True)) 
        # 计算每个类别上的得分
        output = self.map4(u4)

        return output
if __name__ == '__main__':
    x=torch.rand([1, 1, 32, 64, 64]).cuda()
    # net = DeformConvPack_d(1, 32, kernel_size=3, stride=1, padding=1, dimension='HW').cuda()
    net=Our_3dUNet(num_classes=2, in_channel=1).cuda()
    print(net(x).shape)