import torch
import torch.nn as nn
import torch.nn.functional as F
from models.S3_DSConv import DCN_Conv

class SingleConv3DBlock_dsc(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size):
        super().__init__()
        self.block = DCN_Conv(in_planes, out_planes, kernel_size, 1.0, 0, True, 'cuda')

    def forward(self, x):
        y = self.block(x)

        return y

class UNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=2):
        super(UNet, self).__init__()
        self.encoder1 = nn.Sequential(nn.Conv3d(in_channel, 32, 3, stride=1, padding=1),
                                      nn.BatchNorm3d(32))  # b, 16, 10, 10
        self.encoder2 = nn.Sequential(nn.Conv3d(32, 64, 3, stride=1, padding=1),
                                      nn.BatchNorm3d(64))
        self.encoder3 = nn.Sequential(nn.Conv3d(64, 128, 3, stride=1, padding=1),
                                      nn.BatchNorm3d(128))
        self.encoder4 = nn.Sequential(SingleConv3DBlock_dsc(128, 256, 3),
                                      nn.BatchNorm3d(256))
        self.decoder2 = nn.Sequential(nn.Conv3d(256, 128, 3, stride=1, padding=1),
                                      nn.BatchNorm3d(128))
        self.decoder3 = nn.Sequential(nn.Conv3d(128, 64, 3, stride=1, padding=1),
                                      nn.BatchNorm3d(64))
        self.decoder4 = nn.Sequential(nn.Conv3d(64, 32, 3, stride=1, padding=1),
                                      nn.BatchNorm3d(32))
        self.decoder5 = nn.Sequential(nn.Conv3d(32, 2, 3, stride=1, padding=1),
                                      nn.BatchNorm3d(2))

        self.map4 = nn.Sequential(
            nn.Conv3d(2, out_channel, 1, 1),
            nn.Upsample(scale_factor=(1, 1, 1), mode='trilinear', align_corners=True),
            nn.Softmax(dim=1)
        )

        # 128*128 尺度下的映射
        self.map3 = nn.Sequential(
            nn.Conv3d(64, out_channel, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear', align_corners=True),
            nn.Softmax(dim=1)
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(128, out_channel, 1, 1),
            nn.Upsample(scale_factor=(8, 8, 8), mode='trilinear', align_corners=True),
            nn.Softmax(dim=1)
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(256, out_channel, 1, 1),  # （n + 2p -f）/s + 1
            nn.Upsample(scale_factor=(16, 16, 16), mode='trilinear', align_corners=True),
            nn.Softmax(dim=1)  # [1, 3, 32, 128, 128]-->dim = 1:args.labels = 3,放缩labels之和为 1
        )

    def forward(self, x):  # (batch, channel, Depth, Height, Width)：3D卷积5维tensor，x = [batch, 1, 32, 128, 128]
        out = F.relu(F.max_pool3d(self.encoder1(x), 2, 2))  # dilation=1-->普通卷积
        t1 = out  # [6, 32, 16, 64, 64], t1为了torch.add()
        out = F.relu(F.max_pool3d(self.encoder2(out), 2, 2))
        t2 = out  # [6, 64, 8, 32, 32]
        out = F.relu(F.max_pool3d(self.encoder3(out), 2, 2))
        t3 = out  # [6, 128, 4, 16, 16]
        out = F.relu(F.max_pool3d(self.encoder4(out), 2, 2))  # [6, 256, 2, 8, 8], out为了深监督不同特征层输出以及继续上采样
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True))  # [6, 128, 4, 16, 16]
        out = torch.add(out, t3)  # 维度相同,数值相加,3D为相加，2D为拼接，FCN也为像素相加。concat操作,跳层连接   # [1, 128, 4, 16, 16]
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True))   # [1, 64, 8, 32, 32]
        out = torch.add(out, t2)  # [1, 64, 8, 32, 32]
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True))  # [1, 32, 16, 64, 64]
        out = torch.add(out, t1)
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=(2, 2, 2), mode='trilinear', align_corners=True))  # [1, 2, 32, 128, 128]
        output4 = self.map4(out)  # [1, 3, 32, 128, 128]

        return output4  # val
