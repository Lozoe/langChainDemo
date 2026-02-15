import torch
import torch.nn as nn


# 定义神经网络模型（与训练时相同的结构）
class HousingNet(nn.Module):
    def __init__(self, input_size, hidden_size=10):
        super(HousingNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, 1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def preview_model(model_path='best_model.pth'):
    """预览模型信息"""
    
    print("=" * 60)
    print("PyTorch 模型预览")
    print("=" * 60)
    
    # 1. 加载模型权重
    print(f"\n📁 加载模型文件: {model_path}")
    state_dict = torch.load(model_path, map_location='cpu')
    
    # 2. 显示模型层信息
    print("\n🏗️  模型结构信息:")
    print("-" * 60)
    for layer_name, params in state_dict.items():
        print(f"层名称: {layer_name}")
        print(f"  形状: {params.shape}")
        print(f"  参数数量: {params.numel()}")
        print(f"  数据类型: {params.dtype}")
        print()
    
    # 3. 计算总参数量
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"📊 总参数数量: {total_params:,}")
    
    # 4. 显示部分权重值
    print("\n🔍 权重示例 (fc1.weight 的前5行前5列):")
    print("-" * 60)
    fc1_weight = state_dict['fc1.weight']
    print(fc1_weight[:5, :5])
    
    print("\n🔍 偏置示例 (fc1.bias):")
    print("-" * 60)
    print(state_dict['fc1.bias'])
    
    # 5. 重新构建完整模型
    print("\n🤖 重建完整模型:")
    print("-" * 60)
    input_size = state_dict['fc1.weight'].shape[1]  # 从权重形状推断输入大小
    hidden_size = state_dict['fc1.weight'].shape[0]  # 隐藏层大小
    
    model = HousingNet(input_size=input_size, hidden_size=hidden_size)
    model.load_state_dict(state_dict)
    model.eval()
    
    print(model)
    
    # 6. 统计信息
    print("\n📈 权重统计信息:")
    print("-" * 60)
    for layer_name, params in state_dict.items():
        print(f"{layer_name}:")
        print(f"  最小值: {params.min().item():.6f}")
        print(f"  最大值: {params.max().item():.6f}")
        print(f"  均值: {params.mean().item():.6f}")
        print(f"  标准差: {params.std().item():.6f}")
        print()
    
    # 7. 文件大小
    import os
    file_size = os.path.getsize(model_path)
    print(f"💾 文件大小: {file_size:,} bytes ({file_size/1024:.2f} KB)")
    
    # 8. 使用示例
    print("\n💡 使用模型进行预测的示例:")
    print("-" * 60)
    print("```python")
    print("# 加载模型")
    print("model = HousingNet(input_size=13, hidden_size=10)")
    print("model.load_state_dict(torch.load('best_model.pth'))")
    print("model.eval()")
    print()
    print("# 预测（输入需要先标准化）")
    print("with torch.no_grad():")
    print("    input_data = torch.FloatTensor([[...]])  # 13个特征")
    print("    prediction = model(input_data)")
    print("    print(f'预测房价: {prediction.item()}')")
    print("```")
    
    print("\n" + "=" * 60)
    print("预览完成！")
    print("=" * 60)
    
    return model


if __name__ == '__main__':
    model = preview_model('best_model.pth')
