# This Python file uses the following encoding: gbk
"""
��ά�������ݷ���Ƚ�����ӻ�
ʵ�ֹ��ܣ�
1. ������ά�����Կɷ����ݼ�
2. �ȽϾ�������AdaBoost����ͬ�˺���SVM�ķ�������
3. ���ӻ�RBF��SVM�ľ��߽߱�
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn import svm


# ������ά�������ݼ�
def make_moons_3d(n_samples=500, noise=0.1):
    """
    ������ά�����Կɷ����ݼ�
    ������
        n_samples: ÿ������������
        noise: ������׼��
    ���أ�
        X: ��ά�������� (n_samples*2, 3)
        y: ����ǩ (0 �� 1)
    """
    t = np.linspace(0, 2 * np.pi, n_samples)
    x = 1.5 * np.cos(t)
    y = np.sin(t)
    z = np.sin(2 * t)  # ��ӵ���ά�����Ա仯

    # ƴ��������������
    X = np.vstack([np.column_stack([x, y, z]),
                   np.column_stack([-x, y - 1, -z])])  # �ڶ��������о���任
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # ��Ӹ�˹����
    X += np.random.normal(scale=noise, size=X.shape)
    return X, y


# �޸���������Ϊ�������� (ԭѭ��������)
def evaluate_classifier(classifier_func, X, y, X_test, y_test):
    """�����������ڲ��Լ��ϵ�׼ȷ��"""
    return classifier_func(X, y, X_test, y_test)


# ������������
def decision_tree_classifier(X, y, X_test, y_test):
    """Ĭ�ϲ����ľ�����������"""
    clf = DecisionTreeClassifier(random_state=42)  # �̶�������ӱ�֤�ɸ���
    clf.fit(X, y)
    return accuracy_score(y_test, clf.predict(X_test))


# AdaBoost���ɷ�����
def adaboost_classifier(X, y, X_test, y_test):
    """AdaBoost��ǿ�ľ�����������"""
    ada_clf = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3),  # ��������������3
        n_estimators=200,  # ��������
        learning_rate=1,  # ѧϰ��
        algorithm='SAMME',  # ��������ɢ����ǿ
        random_state=42)
    ada_clf.fit(X, y)
    return accuracy_score(y_test, ada_clf.predict(X_test))


# SVM������
def svm_classifier(X, y, X_test, y_test, kernel='rbf'):
    """֧�������������� (֧�ֲ�ͬ�˺���)"""
    clf = svm.SVC(
        kernel=kernel,
        C=1.0,  # ���򻯲���
        gamma='scale',  # ���gamma�������⾯��
        random_state=42)
    clf.fit(X, y)
    return accuracy_score(y_test, clf.predict(X_test))


if __name__ == "__main__":
    # �������ݼ�
    X, y = make_moons_3d(n_samples=1000, noise=0.2)  # ѵ����
    X_test, y_test = make_moons_3d(n_samples=500, noise=0.2)  # �������Լ�

    # �������Ƚ��б�
    classifiers = [
        ("������������", decision_tree_classifier),
        ("AdaBoost+������������", adaboost_classifier),
        ("SVM(���Ժ�)������", lambda X, y, Xt, yt: svm_classifier(X, y, Xt, yt, 'linear')),
        ("SVM(RBF��)������", lambda X, y, Xt, yt: svm_classifier(X, y, Xt, yt, 'rbf')),
        ("SVM(����ʽ��)������", lambda X, y, Xt, yt: svm_classifier(X, y, Xt, yt, 'poly')),
        ("SVM(sigmoid��)������", lambda X, y, Xt, yt: svm_classifier(X, y, Xt, yt, 'sigmoid'))
    ]

    # ������������
    print("���������ܱȽϣ�")
    for name, clf in classifiers:
        accuracy = evaluate_classifier(clf, X, y, X_test, y_test)
        print(f"{name.ljust(20)} ׼ȷ��: {accuracy:.4f}")

    # %% ��ά���ӻ�
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # ����ԭʼ���ݵ�
    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap='viridis',
                         marker='o', s=20, edgecolor='k')
    legend1 = ax.legend(*scatter.legend_elements(), title="Classes")
    ax.add_artist(legend1)

    # ���ɾ��߽߱����� (���ͷֱ����������ٶ�)
    grid_res = 30  # ԭΪ50�����ͷֱ���
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    z_min, z_max = X[:, 2].min() - 1, X[:, 2].max() + 1

    xx, yy, zz = np.meshgrid(
        np.linspace(x_min, x_max, grid_res),
        np.linspace(y_min, y_max, grid_res),
        np.linspace(z_min, z_max, grid_res)
    )

    # ѵ��SVMģ�Ͳ�Ԥ��
    svm_model = svm.SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    svm_model.fit(X, y)
    grid_predictions = svm_model.predict(np.c_[xx.ravel(), yy.ravel(), zz.ravel()])

    # ���ӻ����߽߱� (ʹ��͸���������ɶ���)
    ax.scatter(xx.ravel(), yy.ravel(), zz.ravel(),
               c=grid_predictions,
               cmap='viridis',
               marker='.',
               alpha=0.3,  # ���͸����
               s=1)  # ��С���С

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('3D Make Moons with SVM Classification')
    plt.show()