import utils.ghostnet_data_loader as ghostnet_data_loader
from ghost_model import GhostNet
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import time
import os
import matplotlib.pyplot as plt
import json
import pandas as pd


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """评估模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    accuracy = 100. * correct / total
    avg_loss = running_loss / len(test_loader)
    return avg_loss, accuracy


def save_training_plots(results, save_dir):
    """保存训练过程可视化图像"""
    plt.figure(figsize=(15, 10))

    # 1. 训练和测试损失曲线
    plt.subplot(2, 3, 1)
    plt.plot(results['train_loss'], label='Train Loss')
    plt.plot(results['test_loss'], label='Test Loss')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # 2. 训练和测试准确率曲线
    plt.subplot(2, 3, 2)
    plt.plot(results['train_acc'], label='Train Accuracy')
    plt.plot(results['test_acc'], label='Test Accuracy')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)

    # 3. 学习率变化曲线
    plt.subplot(2, 3, 3)
    plt.plot(results['learning_rate'], label='Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend()
    plt.grid(True)

    # 4. 训练时间分布
    plt.subplot(2, 3, 4)
    plt.plot(results['epoch_time'], label='Epoch Time')
    plt.title('Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.grid(True)

    # 5. 训练集性能对比
    plt.subplot(2, 3, 5)
    epochs = range(1, len(results['train_loss']) + 1)
    plt.plot(epochs, results['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, results['train_acc'], 'r-', label='Train Acc')
    plt.title('Training Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    # 6. 测试集性能对比
    plt.subplot(2, 3, 6)
    plt.plot(epochs, results['test_loss'], 'b-', label='Test Loss')
    plt.plot(epochs, results['test_acc'], 'r-', label='Test Acc')
    plt.title('Test Performance')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_training_results(results, save_dir):
    """保存训练结果到文件"""
    # 保存为JSON
    with open(os.path.join(save_dir, 'training_results.json'), 'w') as f:
        json.dump(results, f, indent=4)

    # 保存为CSV
    df = pd.DataFrame({
        'epoch': range(1, len(results['train_loss']) + 1),
        'train_loss': results['train_loss'],
        'train_acc': results['train_acc'],
        'test_loss': results['test_loss'],
        'test_acc': results['test_acc'],
        'learning_rate': results['learning_rate'],
        'epoch_time': results['epoch_time']
    })
    df.to_csv(os.path.join(save_dir, 'training_results.csv'), index=False)


if __name__ == '__main__':
    # 超参数设置
    batch_size = 128
    learning_rate = 0.1
    epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建保存目录
    save_dir = './ghost_model_checkpoints'
    result_dir = './result'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)

    model_path = os.path.join(save_dir, 'best_ghostnet_cifar10.pth')

    print(f"使用设备: {device}")

    # 数据加载
    print("加载CIFAR-10数据集...")
    train_loader, test_loader = ghostnet_data_loader.get_cifar10_dataloaders(batch_size=batch_size)

    # 创建模型 - 修改输入通道为3
    model = GhostNet(num_classes=10)
    # 修改第一层卷积输入通道从1改为3
    model.conv1[0] = nn.Conv2d(3, 16, 3, 1, 1, bias=False)

    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate,
                          momentum=0.9, weight_decay=5e-4)

    # 学习率调度器
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    print(f"开始训练，总epoch数: {epochs}")
    print(f"模型参数量: {sum(p.numel() for p in model.parameters())}")

    # 初始化结果记录
    training_results = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'learning_rate': [],
        'epoch_time': []
    }

    best_acc = 0
    for epoch in range(epochs):
        start_time = time.time()

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # 评估
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        current_lr = scheduler.get_last_lr()[0]

        epoch_time = time.time() - start_time

        # 记录结果
        training_results['train_loss'].append(train_loss)
        training_results['train_acc'].append(train_acc)
        training_results['test_loss'].append(test_loss)
        training_results['test_acc'].append(test_acc)
        training_results['learning_rate'].append(current_lr)
        training_results['epoch_time'].append(epoch_time)

        # 保存最佳模型到指定目录
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), model_path)

        print(f'Epoch: {epoch + 1:03d} | Time: {epoch_time:.2f}s | '
              f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | '
              f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}% | '
              f'LR: {current_lr:.6f}')

    print(f'训练完成！最佳测试准确率: {best_acc:.2f}%')

    # 保存训练结果和可视化图像
    print("保存训练结果和可视化图像...")
    save_training_results(training_results, result_dir)
    save_training_plots(training_results, result_dir)

    # 最终模型测试
    print("\n最终模型测试:")
    model.load_state_dict(torch.load(model_path))
    final_test_loss, final_test_acc = evaluate(model, test_loader, criterion, device)
    print(f'最终测试准确率: {final_test_acc:.2f}%')

    # 添加最终结果到训练记录
    final_results = {
        'best_accuracy': best_acc,
        'final_accuracy': final_test_acc,
        'final_loss': final_test_loss,
        'total_epochs': epochs,
        'model_parameters': sum(p.numel() for p in model.parameters())
    }

    with open(os.path.join(result_dir, 'final_results.json'), 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"所有结果已保存到 {result_dir} 目录")
