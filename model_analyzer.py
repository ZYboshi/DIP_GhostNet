import torch
import torch.nn as nn
from thop import profile
import os


def analyze_model(model, input_size=(1, 3, 32, 32), model_name="GhostNet"):
    """分析模型指标"""

    # 计算模型大小
    def calculate_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        return size_all_mb

    # 计算FLOPs和参数
    input_tensor = torch.randn(input_size)
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    model_size = calculate_model_size(model)

    print(f"\n=== {model_name} 模型分析 ===")
    print(f"模型大小: {model_size:.2f} MB")
    print(f"参数量: {params:,}")
    print(f"FLOPs: {flops:,}")

    return {
        'model_size_mb': model_size,
        'params': params,
        'flops': flops
    }


if __name__ == '__main__':
    from ghost_model import GhostNet

    # 创建模型
    model = GhostNet(num_classes=10)
    model.conv1[0] = nn.Conv2d(3, 16, 3, 1, 1, bias=False)

    metrics = analyze_model(model)

    # 保存指标
    save_dir = './ghost_model'
    os.makedirs(save_dir, exist_ok=True)
    metrics_path = os.path.join(save_dir, 'model_analysis.txt')

    with open(metrics_path, 'w') as f:
        f.write("模型分析报告\n")
        f.write("============\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
