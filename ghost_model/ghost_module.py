import torch
import torch.nn as nn

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        # 主要卷积产生的特征图通道数
        init_channels = int(oup / ratio)
        # 廉价卷积产生的特征图通道数
        new_channels = oup - init_channels

        # 主要卷积部分：一个普通的 1x1 卷积
        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        # 廉价操作部分：一个深度可分离卷积
        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size//2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        # 将主要卷积和廉价卷积的输出在通道维度上拼接
        out = torch.cat([x1, x2], dim=1)
        return out
