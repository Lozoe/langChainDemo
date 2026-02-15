import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# 设置随机种子，确保结果可复现
torch.manual_seed(42)
np.random.seed(42)


# 1. 加载数据
def load_data(file_path):
    """加载房价数据集"""
    data = pd.read_csv(file_path, sep=r'\s+', header=None)
    # 分离特征和目标
    X = data.iloc[:, :-1].values  # 前13列为特征
    y = data.iloc[:, -1].values.reshape(-1, 1)  # 最后一列为目标（房价）
    return X, y


# 2. 定义神经网络模型
class HousingNet(nn.Module):
    def __init__(self, input_size, hidden_size=10):
        """
        input_size: 输入特征的数量
        hidden_size: 隐藏层神经元数量（默认10）
        """
        super(HousingNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.relu = nn.ReLU()  # 激活函数
        self.fc2 = nn.Linear(hidden_size, 1)  # 隐藏层到输出层
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 3. 训练函数
def train_model(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # 前向传播
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


# 4. 验证函数
def validate_model(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    
    return total_loss / len(val_loader)


# 主程序
if __name__ == '__main__':
    # 配置参数
    DATA_PATH = './housing.csv'
    HIDDEN_SIZE = 10  # 隐藏层神经元数量
    BATCH_SIZE = 32
    LEARNING_RATE = 0.01
    EPOCHS = 200
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据
    print('正在加载数据...')
    X, y = load_data(DATA_PATH)
    print(f'数据集大小: {X.shape[0]} 样本, {X.shape[1]} 特征')
    
    # 切分数据集：80% 训练，20% 验证
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f'训练集: {X_train.shape[0]} 样本, 验证集: {X_val.shape[0]} 样本')
    
    # 数据标准化
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    
    # 转换为 PyTorch 张量
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False
    )
    
    # 初始化模型
    input_size = X_train.shape[1]
    model = HousingNet(input_size=input_size, hidden_size=HIDDEN_SIZE).to(device)
    print(f'\n模型结构:\n{model}')
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 均方误差损失
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练模型
    print(f'\n开始训练 (共 {EPOCHS} 个epoch)...\n')
    best_val_loss = float('inf')
    
    # 记录损失曲线
    train_losses = []
    val_losses = []
    
    for epoch in range(EPOCHS):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss = validate_model(model, val_loader, criterion, device)
        
        # 记录损失值
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
        
        # 每10个epoch打印一次
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], '
                  f'训练损失: {train_loss:.4f}, '
                  f'验证损失: {val_loss:.4f}')
    
    print(f'\n训练完成！最佳验证损失: {best_val_loss:.4f}')
    print(f'最佳模型已保存至: best_model.pth')
    
    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load('best_model.pth'))
    final_val_loss = validate_model(model, val_loader, criterion, device)
    print(f'\n最终验证集MSE损失: {final_val_loss:.4f}')
    
    # 配置中文字体
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'PingFang SC', 'Heiti TC', 'STHeiti']
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, EPOCHS + 1), train_losses, label='训练损失', linewidth=2)
    plt.plot(range(1, EPOCHS + 1), val_losses, label='验证损失', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.title('训练和验证损失曲线', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('loss_curve.png', dpi=300, bbox_inches='tight')
    print('\n损失曲线已保存至: loss_curve.png')
    
    # 显示图片
    plt.show()
