# model_metrics.py
import torch
import torch.nn as nn
from thop import profile
from thop import clever_format
import time
import psutil
import os


def calculate_model_complexity(model, input_size=(1, 3, 224, 224)):
    """
    计算模型的参数量和计算量
    """
    try:
        # 创建随机输入
        input_tensor = torch.randn(input_size).to(next(model.parameters()).device)

        # 使用thop计算FLOPs和参数量
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)

        # === 修复：直接使用原始数值，不格式化 ===
        # 转换为M和G单位
        params_m = params / 1e6
        flops_g = flops / 1e9

        return params_m, flops_g
    except Exception as e:
        print(f"Error calculating model complexity: {e}")
        return 0.0, 0.0


def calculate_model_complexity_advanced(model, input_size=(1, 3, 224, 224)):
    """
    高级版本：提供更详细的复杂度分析
    """
    try:
        device = next(model.parameters()).device
        input_tensor = torch.randn(input_size).to(device)

        # 计算FLOPs和参数量
        flops, params = profile(model, inputs=(input_tensor,), verbose=False)

        # 手动计算每个卷积层的复杂度
        layer_details = []
        total_flops = 0
        total_params = 0

        # 遍历模型层
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d):
                # 计算该卷积层的FLOPs
                kernel_size = layer.kernel_size
                in_channels = layer.in_channels
                out_channels = layer.out_channels

        params_m = params / 1e6
        flops_g = flops / 1e9

        return params_m, flops_g, layer_details
    except Exception as e:
        print(f"Error in advanced complexity calculation: {e}")
        return 0.0, 0.0, []


def measure_inference_speed(model, input_size=(1, 3, 224, 224), num_runs=100):
    """
    测量模型推理速度
    """
    try:
        device = next(model.parameters()).device
        input_tensor = torch.randn(input_size).to(device)

        # GPU预热
        for _ in range(10):
            _ = model(input_tensor)

        # 测量推理时间
        start_time = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                _ = model(input_tensor)

        # 同步GPU操作
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.time()

        # 计算平均推理时间
        avg_time_ms = (end_time - start_time) * 1000 / num_runs
        throughput_imgs = 1000 / avg_time_ms  # 图像/秒

        return avg_time_ms, throughput_imgs
    except Exception as e:
        print(f"Error measuring inference speed: {e}")
        return 0.0, 0.0


def get_memory_usage():
    """
    获取当前内存使用情况
    """
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    return memory_mb


def calculate_accuracy_metrics(results):
    """
    计算Top-1和Top-5准确率
    """
    total_samples = len(results)
    if total_samples == 0:
        return 0.0, 0.0

    correct_top1 = 0
    correct_top5 = 0

    for result in results:
        actual_class = result['actual_class']
        predicted_class = result['predicted_class']

        # Top-1准确率
        if predicted_class == actual_class:
            correct_top1 += 1

    top1_accuracy = (correct_top1 / total_samples) * 100

    return top1_accuracy, 0.0  # 暂时返回0.0作为Top-5准确率


