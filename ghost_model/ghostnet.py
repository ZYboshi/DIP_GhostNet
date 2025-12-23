import torch
import torch.nn as nn
from .ghost_bottleneck import GhostBottleneck

class GhostNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GhostNet, self).__init__()

        # 初始卷积层，将 1 通道（灰度图）扩展到 16 通道
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )

        # 构建瓶颈层序列
        self.blocks = nn.Sequential(
            # in_ch, mid_ch, out_ch, dw_kernel, stride
            GhostBottleneck(16, 16, 16, 3, 1),
            GhostBottleneck(16, 48, 24, 3, 1),
            GhostBottleneck(24, 72, 24, 3, 1),
            GhostBottleneck(24, 72, 40, 3, 2),  # 下采样
            GhostBottleneck(40, 120, 40, 3, 1),
            GhostBottleneck(40, 240, 80, 3, 2), # 下采样
            GhostBottleneck(80, 184, 80, 3, 1),
            GhostBottleneck(80, 184, 80, 3, 1),
            GhostBottleneck(80, 480, 112, 3, 1),
            GhostBottleneck(112, 672, 112, 3, 1),
            GhostBottleneck(112, 672, 160, 3, 1),
            GhostBottleneck(160, 960, 160, 3, 1),
            GhostBottleneck(160, 960, 160, 3, 1),
        )

        # 最后的全局平均池化和全连接层
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv2 = nn.Sequential(
            nn.Conv2d(160, 960, 1, bias=False),
            nn.BatchNorm2d(960),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Linear(960, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 可选：提供一个便捷的创建函数
def ghostnet(num_classes=10, **kwargs):
    return GhostNet(num_classes=num_classes, **kwargs)
