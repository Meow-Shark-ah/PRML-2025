# -*- coding: utf-8 -*-
### 导入必要的库
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error


### 数据预处理函数
def preprocess_data(df, scaler=None, is_training=True):
    """预处理数据用于LSTM模型

    ### 功能说明：
    - 数据清洗：处理缺失值（当前仅检查未处理）
    - 特征工程：风向编码（分类转数值）
    - 标准化处理：MinMax归一化
    - 特征选择：保留8个关键特征

    参数说明：
        df: 原始数据框（需包含date,wnd_dir等字段）
        scaler: 已拟合的标准化器（测试时传入）
        is_training: 标记训练/测试模式

    返回：
        处理后的数据和标准化器（训练模式）
    """
    # 创建副本避免修改原始数据
    df_scaled = df.copy()

    # 风向编码（分类特征数字化）
    mapping = {'NE': 0, 'SE': 1, 'NW': 2, 'cv': 3}
    df_scaled['wnd_dir'] = df_scaled['wnd_dir'].map(mapping)

    # 日期处理（转换为时间序列索引）
    if 'date' in df_scaled.columns:
        df_scaled['date'] = pd.to_datetime(df_scaled['date'])
        df_scaled.set_index('date', inplace=True)

    # 特征选择（8个关键气象特征）
    columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    df_scaled = df_scaled[columns]

    # 数据标准化（保证训练测试一致性）
    if is_training:
        scaler = MinMaxScaler()
        df_scaled[columns] = scaler.fit_transform(df_scaled[columns])
        return df_scaled, scaler
    else:
        if scaler is None:
            raise ValueError("测试数据必须提供训练时的scaler")
        df_scaled[columns] = scaler.transform(df_scaled[columns])
        return df_scaled


### 时间序列样本生成函数
def create_sequences(data, n_past, n_future):
    """创建LSTM所需的序列数据

    ### 参数说明：
        data: 标准化后的numpy数组
        n_past: 历史时间步长（如10小时）
        n_future: 预测时间步长（如1小时）

    ### 生成逻辑：
        X格式：[样本数, 时间步长, 特征数]
        y格式：[样本数, 预测目标]
    """
    X, y = [], []
    # 滑动窗口生成序列（边界处理：确保不越界）
    for i in range(n_past, len(data) - n_future + 1):
        X.append(data[i - n_past:i, 1:])  # 取特征列（排除目标列pollution）
        y.append(data[i + n_future - 1:i + n_future, 0])  # 取目标值（pollution）
    return np.array(X), np.array(y)


### LSTM模型定义
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers, dropout_rate):
        super(LSTMModel, self).__init__()
        ### 模型参数
        self.input_size = input_size  # 输入特征维度（7个气象特征）
        self.hidden_size = hidden_size  # 隐层单元数（64）
        self.output_size = output_size  # 输出维度（1）
        self.num_layers = num_layers  # LSTM层数（2）
        self.dropout = nn.Dropout(dropout_rate)  # 防止过拟合

        ### 自动选择计算设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        ### 网络层定义
        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,  # 输入格式(batch, seq, feature)
            dropout=dropout_rate if num_layers > 1 else 0
        )
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)  # 全连接输出层

    def forward(self, x):
        ### 初始化隐状态（全零初始化）
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        ### LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))  # out shape: (batch_size, seq_length, hidden_size)

        ### 提取最后时间步输出
        out = out[:, -1, :]  # 取最后一个时间步的输出

        ### 正则化处理
        out = self.dropout(out)

        ### 全连接层输出
        out = self.fc(out)
        return out


### 单epoch训练函数
def train_epoch(net, train_iter, optimizer, loss_fn):
    """执行一个epoch的训练

    ### 流程说明：
        1. 切换训练模式
        2. 遍历数据加载器
        3. 前向传播计算损失
        4. 反向传播更新参数
        5. 收集指标数据
    """
    net.train()
    train_loss = []
    predictions = []
    targets = []

    loop = tqdm(train_iter, desc='Train')
    device = next(net.parameters()).device  # 自动获取模型所在设备

    for X, y in loop:
        ### 数据迁移到对应设备
        X, y = X.to(device), y.to(device)

        ### 梯度清零
        optimizer.zero_grad()

        ### 前向传播
        y_hat = net(X)

        ### 计算损失
        loss = loss_fn(y_hat, y)
        train_loss.append(loss.item())

        ### 收集预测值和真实值
        predictions.extend(y_hat.cpu().detach().numpy())
        targets.extend(y.cpu().detach().numpy())

        ### 反向传播
        loss.backward()
        optimizer.step()

    ### 计算全局指标
    rmse = sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    return sum(train_loss) / len(train_loss), rmse, mae


### 模型评估函数
@torch.no_grad()
def eval_model(net, test_iter, loss_fn):
    """模型评估（测试模式）"""
    net.eval()
    test_loss = []
    predictions = []
    targets = []

    loop = tqdm(test_iter, desc='Test')
    device = next(net.parameters()).device

    for X, y in loop:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        test_loss.append(loss.item())

        predictions.extend(y_hat.cpu().detach().numpy())
        targets.extend(y.cpu().detach().numpy())

    rmse = sqrt(mean_squared_error(targets, predictions))
    mae = mean_absolute_error(targets, predictions)
    return sum(test_loss) / len(test_loss), rmse, mae


### 模型训练主函数
def train_model(model, train_loader, test_loader, epochs=20, patience=3, lr=0.001):
    """模型训练总控函数

    ### 核心功能：
        - 动态学习率调整
        - 早停机制
        - 模型保存
        - 训练过程监控
    """
    ### 初始化训练组件
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

    ### 训练记录器
    train_losses, train_rmses, train_maes = [], [], []
    test_losses, test_rmses, test_maes = [], [], []

    ### 早停机制参数
    best_rmse = float('inf')
    patience_counter = 0
    best_model_state = None

    ### 训练循环
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        ### 训练与评估
        train_loss, train_rmse, train_mae = train_epoch(model, train_loader, optimizer, loss_fn)
        test_loss, test_rmse, test_mae = eval_model(model, test_loader, loss_fn)

        ### 学习率调整
        scheduler.step(test_loss)

        ### 记录训练时间
        train_time = time.time() - start_time

        ### 存储指标
        train_losses.append(train_loss)
        train_rmses.append(train_rmse)
        train_maes.append(train_mae)
        test_losses.append(test_loss)
        test_rmses.append(test_rmse)
        test_maes.append(test_mae)

        ### 打印训练进度
        print(f"Epoch: {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.6f} | Test Loss: {test_loss:.6f} | "
              f"Train RMSE: {train_rmse:.6f} | Test RMSE: {test_rmse:.6f} | "
              f"Train MAE: {train_mae:.6f} | Test MAE: {test_mae:.6f} | "
              f"Time: {train_time:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")

        ### 早停判断
        if test_rmse < best_rmse:
            best_rmse = test_rmse
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    ### 最终处理
    print(f"Best Test RMSE: {best_rmse:.6f}")
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, 'best_lstm_model.pth')
    print("Best model saved as 'best_lstm_model.pth'")

    return {
        'train_losses': train_losses,
        'train_rmses': train_rmses,
        'train_maes': train_maes,
        'test_losses': test_losses,
        'test_rmses': test_rmses,
        'test_maes': test_maes
    }


### 训练曲线可视化函数
def plot_learning_curves(history):
    """绘制训练过程指标曲线"""
    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))

    ### 损失曲线
    axs[0].plot(history['train_losses'], label='Train Loss', color='blue')
    axs[0].plot(history['test_losses'], label='Test Loss', color='green')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss (MSE)')
    axs[0].set_title('Loss Curves')
    axs[0].legend()

    ### RMSE曲线
    axs[1].plot(history['train_rmses'], label='Train RMSE', color='blue')
    axs[1].plot(history['test_rmses'], label='Test RMSE', color='green')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('RMSE')
    axs[1].set_title('RMSE Curves')
    axs[1].legend()

    ### MAE曲线
    axs[2].plot(history['train_maes'], label='Train MAE', color='blue')
    axs[2].plot(history['test_maes'], label='Test MAE', color='green')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('MAE')
    axs[2].set_title('MAE Curves')
    axs[2].legend()

    plt.tight_layout()
    plt.savefig('learning_curves.png')
    plt.show()


### 主执行函数
def main():
    """主流程控制函数"""
    ### 设置随机种子（保证可重复性）
    np.random.seed(42)
    torch.manual_seed(42)

    ### 自动选择计算设备
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    ### 数据加载与检查
    print("Loading data...")
    df_train = pd.read_csv('LSTM-Multivariate_pollution.csv')
    df_test = pd.read_csv('pollution_test_data1.csv.xls')

    print("\nMissing values in training data:")
    print(df_train.isnull().sum())
    print("\nMissing values in test data:")
    print(df_test.isnull().sum())

    print("\nTraining data summary:")
    print(df_train.describe())

    ### 数据预处理
    print("\nPreprocessing data...")
    df_train_scaled, scaler = preprocess_data(df_train, is_training=True)
    df_test_scaled = preprocess_data(df_test, scaler=scaler, is_training=False)

    ### 转换为numpy数组
    df_train_scaled_np = np.array(df_train_scaled)
    df_test_scaled_np = np.array(df_test_scaled)

    ### 生成时间序列样本
    print("\nCreating sequences...")
    n_past = 10  # 使用前10小时数据
    n_future = 1  # 预测1小时后
    X_train, y_train = create_sequences(df_train_scaled_np, n_past, n_future)
    X_test, y_test = create_sequences(df_test_scaled_np, n_past, n_future)

    print(f'X_train shape: {X_train.shape}, y_train shape: {y_train.shape}')
    print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')

    ### 数据转换与加载
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    ### 模型参数定义
    input_size = X_train.shape[2]  # 输入特征数（7个气象特征）
    hidden_size = 64  # LSTM隐层单元数
    output_size = 1  # 输出维度（预测PM2.5）
    num_layers = 2  # LSTM层数
    dropout_rate = 0.3  # Dropout比例

    ### 模型初始化
    print("\nCreating LSTM model...")
    model = LSTMModel(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        num_layers=num_layers,
        dropout_rate=dropout_rate
    ).to(device)

    print("\nModel Architecture:")
    print(model)
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    ### 模型训练
    print("\nTraining model...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        epochs=30,  # 最大训练轮次
        patience=5,  # 早停耐心值
        lr=0.001  # 初始学习率
    )

    ### 结果可视化
    print("\nPlotting learning curves...")
    plot_learning_curves(history)


if __name__ == '__main__':
    main()