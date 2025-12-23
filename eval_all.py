import json
import torch
import torchvision.transforms as transforms
from PIL import Image
import os
import csv

import args
import models
from utils.model_metrics  import calculate_model_complexity, measure_inference_speed, get_memory_usage, calculate_accuracy_metrics

#返回分类结果
def predict_single_image(model, image_path):
    try:
        # 加载图片
        image = Image.open(image_path).convert('RGB')
        # 定义数据预处理变换
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],   # 归一化
                                std=[0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0)
        # 将输入张量移入GPU
        input_tensor = input_tensor.to(device)

        # 进行推理
        with torch.no_grad():
            output = model(input_tensor)

        # 获取预测结果
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()
        print(f"预测的类别索引: {predicted_class}")
        print(f"最大概率: {probabilities[predicted_class]:.4f}")
        return predicted_class, probabilities[predicted_class].item()


    except Exception as e:
        print(f"Error predicting image: {e}")
        return None, None

def load_model(model_name):
    supported_models = ['ghostnet_100','ghostnet_130','mobilenetv3_small','mobilenetv3_large']

    try:
        if model_name not in supported_models:
            raise ValueError('Model not supported')
        model = models.get_model(model_name=model_name, pretrained=True, num_classes=args.num_classes)
        model = model.to(device)
        model.eval()
        return model
    except ValueError as e:
        print(e)
        exit(1)
# 获得映射表
def load_classes_mapping(json_file = args.json_path):
    with open(json_file,'r') as file:
        class_mapping = json.load(file)

    # 创建一个字典，键为文件名，值为类别编号
    file_to_class = {v[0]: k for k, v in class_mapping.items()}
    return file_to_class


# 在 import 部分之后添加
def analyze_model_metrics(model, model_name):
    """
    分析模型性能指标（不修改原有逻辑）
    """
    print(f"\n📊 开始分析 {model_name} 模型复杂度...")

    # 1. 计算模型复杂度
    params_m, flops_g = calculate_model_complexity(model)
    print(f"  → 参数量: {params_m:.2f}M")
    print(f"  → FLOPs: {flops_g:.2f}G")

    # 2. 测量推理速度
    avg_time_ms, throughput_imgs = measure_inference_speed(model)
    print(f"  → 平均推理时间: {avg_time_ms:.2f}ms")
    print(f"  → 吞吐量: {throughput_imgs:.2f} img/s")

    # 3. 获取内存使用情况
    memory_mb = get_memory_usage()
    print(f"  → 内存使用: {memory_mb:.2f}MB")

    return {
        'params_m': params_m,
        'flops_g': flops_g,
    }




#将预测结果以及
def main(model_name, output_csv='predictions.csv'):
    model = load_model(model_name)
    print('加载模型成功')

    # === 新增：在开始预测前分析模型指标 ===
    metrics = analyze_model_metrics(model, model_name)

    file_to_class = load_classes_mapping()

    #存储image_path于字典中，方便后续寻找
    image_data = {}
    image_dir = os.path.join('dataset','ImageNet-Mini','images')
    for image_folder in os.listdir(image_dir):
        folder_path = os.path.join(image_dir,image_folder)
        image_data[image_folder] = []
        for image_file in os.listdir(folder_path):
            image_path = os.path.join(folder_path,image_file)
            image_data[image_folder].append(image_path)

    results = []
    for image_folder,image_paths in image_data.items():
        #遍历每个图片目录
        for image_path in image_paths:
            predicted_class, probability = predict_single_image(model, image_path)
            if predicted_class is not None:
                actual_class = int(file_to_class.get(image_folder, -1))  # 获取实际类别编号
                results.append({
                    'image_path': image_path,
                    'folder_name': image_folder,
                    'predicted_class': predicted_class,
                    'probability': probability,
                    'actual_class': actual_class
                })

    # 写入CSV文件
    with open(output_csv, mode='w', newline='') as file:
        fieldnames = ['image_path', 'folder_name', 'predicted_class', 'probability', 'actual_class']
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_name = 'ghostnet_100'
    output_csv = './result/predictions_ghostnet.csv'
    main(model_name=model_name, output_csv=output_csv)
    model_name = 'mobilenetv3_small'
    output_csv = './result/mobilenetv3_small.csv'
    main(model_name=model_name, output_csv=output_csv)