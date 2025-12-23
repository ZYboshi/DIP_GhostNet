#针对result做可视化数据
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 支持多种字体回退
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def compare_two_models(file1_path, file1_name, file2_path, file2_name):
    """
    对比两个模型结果的函数

    参数:
    file1_path: 第一个CSV文件路径
    file1_name: 第一个模型名称
    file2_path: 第二个CSV文件路径
    file2_name: 第二个模型名称
    """

    # 读取数据
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)

    # 计算准确率
    acc1 = (df1['predicted_class'] == df1['actual_class']).mean()
    acc2 = (df2['predicted_class'] == df2['actual_class']).mean()

    # 创建对比可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 1. 准确率对比柱状图
    models = [file1_name, file2_name]
    accuracies = [acc1, acc2]

    bars = axes[0, 0].bar(models, accuracies, color=['skyblue', 'lightcoral'], alpha=0.7)
    axes[0, 0].set_title('模型准确率对比')
    axes[0, 0].set_ylabel('准确率')
    axes[0, 0].set_ylim(0, 1)

    # 在柱状图上添加数值标签
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width() / 2., height + 0.02,
                        f'{acc:.2%}', ha='center', va='bottom')

    # 2. 概率分布对比直方图
    axes[0, 1].hist(df1['probability'], bins=20, alpha=0.5, label=file1_name, color='blue')
    axes[0, 1].hist(df2['probability'], bins=20, alpha=0.5, label=file2_name, color='red')
    axes[0, 1].axvline(0.5, color='black', linestyle='--', label='决策边界')
    axes[0, 1].set_title('预测概率分布对比')
    axes[0, 1].set_xlabel('预测概率')
    axes[0, 1].set_ylabel('样本数量')
    axes[0, 1].legend()

    # 3. 概率密度曲线对比
    axes[1, 0].hist(df1['probability'], bins=30, density=True, alpha=0.5,
                    label=file1_name, color='blue')
    axes[1, 0].hist(df2['probability'], bins=30, density=True, alpha=0.5,
                    label=file2_name, color='red')
    axes[1, 0].axvline(0.5, color='black', linestyle='--', label='决策边界')
    axes[1, 0].set_title('概率密度分布对比')
    axes[1, 0].set_xlabel('预测概率')
    axes[1, 0].set_ylabel('密度')
    axes[1, 0].legend()

    # 4. 箱线图对比
    prob_data = [df1['probability'], df2['probability']]
    axes[1, 1].boxplot(prob_data, labels=models)
    axes[1, 1].set_title('预测概率箱线图对比')
    axes[1, 1].set_ylabel('预测概率')

    plt.tight_layout()
    plt.show()

    # 打印对比统计信息
    print("=" * 50)
    print("模型对比统计信息")
    print("=" * 50)
    print(f"{file1_name}:")
    print(f"  准确率: {acc1:.2%}")
    print(f"  平均概率: {df1['probability'].mean():.4f}")
    print(f"  概率标准差: {df1['probability'].std():.4f}")

    print(f"\n{file2_name}:")
    print(f"  准确率: {acc2:.2%}")
    print(f"  平均概率: {df2['probability'].mean():.4f}")
    print(f"  概率标准差: {df2['probability'].std():.4f}")

    print(f"\n准确率差异: {abs(acc1 - acc2):.4f} ({abs(acc1 - acc2) * 100:.2f}%)")


# 使用示例
if __name__ == "__main__":
    # 替换为您的实际文件路径和模型名称
    file1_path = "../result/mobilenetv3_small.csv"  # 第一个文件路径
    file1_name = "mobilenetv3_small"  # 第一个模型名称

    file2_path = "../result/predictions_ghostnet.csv"  # 第二个文件路径
    file2_name = "ghostnet_100"  # 第二个模型名称

    compare_two_models(file1_path, file1_name, file2_path, file2_name)
