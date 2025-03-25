import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文显示和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取数据（假设数据文件中有两个sheet）
DATA_FILE = "Data4Regression.xlsx"  # 常量使用全大写命名
train_data = pd.read_excel(DATA_FILE, sheet_name=0)  # 训练数据
test_data = pd.read_excel(DATA_FILE, sheet_name=1)  # 测试数据

# 提取特征和目标列，转换为numpy数组
x_train = train_data.iloc[:, 0].values  # 训练集特征
y_train = train_data.iloc[:, 1].values  # 训练集目标
x_test = test_data.iloc[:, 0].values  # 测试集特征
y_test = test_data.iloc[:, 1].values  # 测试集目标

# 构建设计矩阵（添加截距项）
X_train = np.c_[np.ones_like(x_train), x_train]  # 使用np.c_进行列合并
X_test = np.c_[np.ones_like(x_test), x_test]


def calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """计算均方误差（Mean Squared Error, MSE）"""
    return np.mean((y_true - y_pred) ** 2)


# ================== 最小二乘法 ==================
def least_squares(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """通过正规方程求解线性回归参数"""
    return np.linalg.inv(X.T @ X) @ (X.T @ y)  # (X^T X)^-1 X^T y


theta_ls = least_squares(X_train, y_train)
y_train_pred_ls = X_train @ theta_ls
y_test_pred_ls = X_test @ theta_ls
mse_train_ls = calculate_mse(y_train, y_train_pred_ls)
mse_test_ls = calculate_mse(y_test, y_test_pred_ls)

print("[最小二乘法]")
print(f"参数估计（截距, 斜率）: {theta_ls.round(4)}")
print(f"训练集 MSE: {mse_train_ls:.4f}")
print(f"测试集 MSE: {mse_test_ls:.4f}\n")


# ================== 梯度下降法 ==================
def gradient_descent(
        X: np.ndarray,
        y: np.ndarray,
        learning_rate: float = 0.01,
        max_iters: int = 1000,
        tol: float = 1e-6
) -> tuple[np.ndarray, list]:
    """
    使用批量梯度下降法求解参数
    参数:
        tol: 当MSE变化小于该值时提前停止
    """
    m, n = X.shape
    theta = np.zeros(n)
    mse_history = []

    for _ in range(max_iters):
        y_pred = X @ theta
        gradient = (1 / m) * X.T @ (y_pred - y)  # 计算梯度
        theta -= learning_rate * gradient  # 参数更新

        current_mse = calculate_mse(y, y_pred)
        mse_history.append(current_mse)

        # 早停机制：当MSE变化小于阈值时停止
        if len(mse_history) > 1 and abs(mse_history[-2] - current_mse) < tol:
            break

    return theta, mse_history


theta_gd, gd_loss = gradient_descent(X_train, y_train)
y_train_pred_gd = X_train @ theta_gd
y_test_pred_gd = X_test @ theta_gd
mse_train_gd = calculate_mse(y_train, y_train_pred_gd)
mse_test_gd = calculate_mse(y_test, y_test_pred_gd)

print("[梯度下降法]")
print(f"参数估计（截距, 斜率）: {theta_gd.round(4)}")
print(f"训练集 MSE: {mse_train_gd:.4f}")
print(f"测试集 MSE: {mse_test_gd:.4f}")
print(f"迭代次数: {len(gd_loss)}，最终MSE: {gd_loss[-1]:.4f}\n")


# ================== 牛顿法 ==================
def newton_method(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, list]:
    """牛顿法（适用于线性回归的二次情况，一次迭代即可收敛）"""
    m, n = X.shape
    theta = np.zeros(n)

    # 预先计算Hessian矩阵的逆（对于线性回归是常数矩阵）
    H_inv = np.linalg.inv((1 / m) * (X.T @ X))  # (X^T X/m)^-1

    # 参数更新：theta = theta - H^{-1} * gradient
    gradient = (1 / m) * X.T @ (X @ theta - y)
    theta -= H_inv @ gradient

    # 由于一次迭代即可收敛，直接计算MSE
    mse = calculate_mse(y, X @ theta)
    return theta, [mse]


theta_nt, nt_loss = newton_method(X_train, y_train)
y_train_pred_nt = X_train @ theta_nt
y_test_pred_nt = X_test @ theta_nt
mse_train_nt = calculate_mse(y_train, y_train_pred_nt)
mse_test_nt = calculate_mse(y_test, y_test_pred_nt)

print("[牛顿法]")
print(f"参数估计（截距, 斜率）: {theta_nt.round(4)}")
print(f"训练集 MSE: {mse_train_nt:.4f}")
print(f"测试集 MSE: {mse_test_nt:.4f}")
print(f"迭代次数: {len(nt_loss)}，最终MSE: {nt_loss[-1]:.4f}\n")

# ================== 可视化 ==================
# 创建预测用的均匀分布数据点
x_range = np.linspace(
    min(x_train.min(), x_test.min()),
    max(x_train.max(), x_test.max()),
    300
)
X_range = np.c_[np.ones_like(x_range), x_range]  # 设计矩阵

# 各方法的预测曲线
y_ls = X_range @ theta_ls  # 最小二乘法
y_gd = X_range @ theta_gd  # 梯度下降法
y_nt = X_range @ theta_nt  # 牛顿法

plt.figure(figsize=(10, 6))
plt.scatter(x_train, y_train, c='blue', edgecolor='w', label='训练数据')
plt.scatter(x_test, y_test, c='green', edgecolor='w', label='测试数据')

plt.plot(x_range, y_ls, 'r-', lw=2.5, label=f'最小二乘 (MSE={mse_test_ls:.2f})')
plt.plot(x_range, y_gd, '--', color='orange', lw=2, label=f'梯度下降 (MSE={mse_test_gd:.2f})')
plt.plot(x_range, y_nt, '-.', color='purple', lw=2, label=f'牛顿法 (MSE={mse_test_nt:.2f})')

plt.xlabel("特征 x", fontsize=12)
plt.ylabel("目标 y", fontsize=12)
plt.title("不同回归方法的拟合效果对比", fontsize=14)
plt.legend(loc='upper left', frameon=True)
plt.grid(alpha=0.3)
plt.tight_layout()

# 绘制收敛曲线对比
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(gd_loss, 'b-o', markersize=4, lw=1)
plt.title("梯度下降法收敛过程")
plt.xlabel("迭代次数")
plt.ylabel("训练集MSE")
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(nt_loss, 'm-s', markersize=6)
plt.title("牛顿法收敛过程")
plt.xlabel("迭代次数")
plt.ylabel("训练集MSE")
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()