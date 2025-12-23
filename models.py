"""集中管理所有模型定义"""
import timm
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image

def get_model(model_name, num_classes=1000, pretrained=True):
    """
    一站式获取模型
    Args:
        model_name: ghostnet_100, mobilenetv3_large, shufflenet_v2, 等
        num_classes: 分类数
        pretrained: 是否使用预训练权重
    """
    # 支持的模型列表
    supported_models = {
        'ghostnet_100': 'ghostnet_100',
        'ghostnet_130': 'ghostnet_130',
        'mobilenetv3_small': 'mobilenetv3_small_075',
        'mobilenetv3_large': 'mobilenetv3_large_100',
    }

    if model_name not in supported_models:
        raise ValueError(f"不支持的模型: {model_name}")

    # 直接用timm创建
    model = timm.create_model(
        supported_models[model_name],
        pretrained=pretrained,
        num_classes=num_classes
    )

    return model


# 快捷函数
def ghostnet_100(num_classes=1000, pretrained=True):
    return get_model('ghostnet_100', num_classes, pretrained)


def mobilenetv3_small(num_classes=1000, pretrained=True):
    return get_model('mobilenetv3_small', num_classes, pretrained)

def test():
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    image_path ="./dataset/ImageNet-Mini/images/n01443537/ILSVRC2012_val_00000994.JPEG"

    transform = transforms.Compose([    #列表打包处理步骤
    transforms.Resize(256),             #把图片的短边缩放到256像素，长边按比例缩放
    transforms.CenterCrop(224),         #从图片中心裁切出224x224的区域
    transforms.ToTensor(),              #image.open拿出来的时候是：[0,255]的数值
    transforms.Normalize(mean=[0.485, 0.456, 0.406],   #归一化
                         std=[0.229, 0.224, 0.225])
    ])

    # 加载图片 - image: format , mode  , size  ,数据格式:像素值
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    # 将输入张量移到GPU
    input_tensor = input_tensor.to(device)


    # 加载模型并设置为评估模式
    model = mobilenetv3_small(pretrained=True)
    # 将模型移到GPU
    model = model.to(device)

    model.eval()
    print(f"模型处于评估模式: {not model.training}")
    print(f"输入张量形状: {input_tensor.shape}")
    # 进行推理
    with torch.no_grad():
        output = model(input_tensor)

    # 获取预测结果
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class = torch.argmax(probabilities).item()

    print(f"预测的类别索引: {predicted_class}")
    print(f"最大概率: {probabilities[predicted_class]:.4f}")




def main():
    model = ghostnet_100(pretrained=True)
    model.eval()
    print(f"模型类型: {type(model)}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    # 2. 前向传播测试
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape}")
    # 3. 检查模型状态
    print(f"是否在训练模式: {model.training}")
    # 4. 验证分类类别数
    print(f"输出维度: {output.shape[1]} (应该是 {1000})")
    print("✓ 模型加载成功" if output.shape[1] == 1000 else "✗ 模型加载异常")

if __name__ == '__main__':
    # 测试能否引入模型
    test()