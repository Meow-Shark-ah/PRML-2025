# This Python file uses the following encoding: gbk
"""
三维月亮数据分类比较与可视化
实现功能：
1. 生成三维非线性可分数据集
2. 比较决策树、AdaBoost、不同核函数SVM的分类性能
3. 可视化RBF核SVM的决策边界
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm


# 生成三维月亮数据集
def make_moons_3d(n_samples=500, noise=0.1):
    """
    生成三维非线性可分数据集
    参数：
        n_samples: 每个类别的样本数
        noise: 噪声标准差
    返回：
        X: 三维坐标数据 (n_samples*2, 3)
        y: 类别标签 (0 或 1)
    """
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # 添加第三维周期性变化

    # 拼接两个类别的数据
    X = np.vstack([np.column_stack([x, y, z]),
                   np.column_stack([-x, y - 1, -z])])  # 第二个类别进行镜像变换
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # 添加高斯噪声
    X += np.random.normal(scale=noise, size=X.shape)
    return X, y


# 修改评估函数为单次评估 (原循环无意义)
def evaluate_classifier(classifier_func, X, y, X_test, y_test):
    """评估分类器在测试集上的准确率"""
    return classifier_func(X, y, X_test, y_test)


# 决策树分类器
def decision_tree_classifier(X, y, X_test, y_test):
    """默认参数的决策树分类器"""
    clf = DecisionTreeClassifier(random_state=42)  # 固定随机种子保证可复现
    clf.fit(X, y)
    return accuracy_score(y_test, clf.predict(X_test))


# AdaBoost集成分类器
def adaboost_classifier(X, y, X_test, y_test):
    """AdaBoost增强的决策树分类器"""
    ada_clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),  # 基分类器最大深度3
        n_estimators=200,  # 迭代次数
        learning_rate=1,  # 学习率
        algorithm='SAMME',  # 适用于离散型增强
        random_state=42)
    ada_clf.fit(X, y)
    return accuracy_score(y_test, ada_clf.predict(X_test))


# SVM分类器
def svm_classifier(X, y, X_test, y_test, kernel='rbf'):
    """支持向量机分类器 (支持不同核函数)"""
    clf = svm.SVC(
        kernel=kernel,
        C=1.0,  # 正则化参数
        gamma='scale',  # 添加gamma参数避免警告
        random_state=42)
    clf.fit(X, y)
    return accuracy_score(y_test, clf.predict(X_test))


if __name__ == "__main__":
    # 生成数据集
    X, y = make_moons_3d(n_samples=1000, noise=0.2)  # 训练集
    X_test, y_test = make_moons_3d(n_samples=500, noise=0.2)  # 独立测试集

    # 分类器比较列表
    classifiers = [
        ("决策树分类器", decision_tree_classifier),
        ("AdaBoost+决策树分类器", adaboost_classifier),
        ("SVM(线性核)分类器", lambda X, y, Xt, yt: svm_classifier(X, y, Xt, yt, 'linear')),
        ("SVM(RBF核)分类器", lambda X, y, Xt, yt: svm_classifier(X, y, Xt, yt, 'rbf')),
        ("SVM(多项式核)分类器", lambda X, y, Xt, yt: svm_classifier(X, y, Xt, yt, 'poly')),
        ("SVM(sigmoid核)分类器", lambda X, y, Xt, yt: svm_classifier(X, y, Xt, yt, 'sigmoid'))
    ]

    # 评估并输出结果
    print("分类器性能比较：")
    for name, clf in classifiers:
        accuracy = evaluate_classifier(clf, X, y, X_test, y_test)
        print(f"{name.ljust(20)} 准确率: {accuracy:.4f}")

    # %% 三维可视化
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制原始数据点
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis',
                         marker='o', s=20, edgecolor='k')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)

    # 生成决策边界网格 (降低分辨率以提升速度)
    grid_res = 30  # 原为50，降低分辨率
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

    xx, yy, zz = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
        np.linspace(z_min, z_max, grid_res)
    )

    # 训练SVM模型并预测
    svm_model = svm.SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_model.fit(X, y)
    grid_predictions = svm_model.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])

    # 可视化决策边界 (使用透明度提升可读性)
    ax.scatter(xx.ravel(), yy.ravel(), zz.ravel(),
               c=grid_predictions,
               cmap='viridis',
               marker='.',
               alpha=0.3,  # 添加透明度
               s=1)  # 减小点大小

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Make Moons with SVM Classification')
    plt.show()