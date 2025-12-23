import torch
import torch.nn as nn
from .ghost_module import GhostModule

class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, dw_kernel_size=3, stride=1):
        super(GhostBottleneck, self).__init__()
        self.stride = stride

        # 第一个 Ghost 模块用于扩张通道数
        self.ghost1 = GhostModule(in_channels, mid_channels, relu=True)

        # 如果进行下采样（stride=2），使用深度卷积
        if self.stride > 1:
            self.dw_conv = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size-1)//2, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
            )
        else:
            self.dw_conv = None

        # 第二个 Ghost 模块用于压缩通道数，不使用 ReLU
        self.ghost2 = GhostModule(mid_channels, out_channels, relu=False)

        #  shortcut 连接
        if in_channels == out_channels and stride == 1:
            self.shortcut = nn.Sequential()
        else:
            # 如果输入输出通道数不匹配或需要下采样，使用 1x1 卷积调整
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        residual = x

        x = self.ghost1(x)

        if self.dw_conv is not None:
            x = self.dw_conv(x)

        x = self.ghost2(x)

        # 添加 shortcut
        x += self.shortcut(residual)
        return x
