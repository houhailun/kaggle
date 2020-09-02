#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/9/9 11:42
# Author: Hou hailun

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, train_test_split


def basic_model():
    # 读取数据
    train_data = pd.read_csv("../dataset/train.csv", header=None)
    train_label_data = pd.read_csv("../dataset/trainLabels.csv", header=None)
    test_data = pd.read_csv("../dataset/test.csv", header=None)

    print(plt.style.available)  # look at available plot styles
    plt.style.use('ggplot')

    # print('train shape:', train_data.shape)  # (1000,40)
    # print('test shape:', test_data.shape)    # (9000,40)
    # print('train_label shape:', train_label_data.shape)  # (1000,1)
    # print(train_data.head())
    # print(train_data.info())  # 查看是否有空
    # print(train_data.describe())  # 查看数值型特征分布情况

    X, y = train_data, np.ravel(train_label_data)  # ravel()把多维数组降为一维，返回视图
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 训练：测试 = 7：3

    # Model complexity
    neig = np.arange(1, 25)
    kflod = 10
    train_accuracy = []
    val_accuracy = []
    best_knn = None
    best_acc = 0.0
    # 超参数调参
    for i, k in enumerate(neig):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_accuracy.append(knn.score(X_train, y_train))
        val_accuracy.append(np.mean(cross_val_score(knn, X, y, cv=kflod)))
        if np.mean(cross_val_score(knn, X, y, cv=kflod)) > best_acc:
            best_acc = np.mean(cross_val_score(knn, X, y, cv=kflod))
            best_knn = knn

    # plot
    plt.figure(figsize=[13, 8])
    plt.plot(neig, val_accuracy, label='Validation Accuracy')
    plt.plot(neig, train_accuracy, label='Training Accuracy')
    plt.legend()
    plt.title('k value As Accuracy')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Acuracy')
    plt.xticks(neig)
    plt.show()

    print('Best Accuracy without feature sacling:', best_acc)
    print(best_knn)

    # 保存文件
    test_fill = np.nan_to_num(test_data)  # 把nan转换为0或inf，-inf
    submission = pd.DataFrame(best_knn.predict(test_fill))
    print(submission.shape)
    submission.columns = ['Solution']
    submission['Id'] = np.arange(1, submission.shape[0]+1)
    submission = submission[['Id', 'Solution']]
    print(submission.head())

    submission.to_csv('../dataset/submission_no_normalization.csv', index=False)  # index标识是否写列名称


# 特征工程
def model_with_feature_engineering():
    # 模型缩放
    # StandardScaler: (X-X_mean) / X_std, 均值为0，方差为1
    # MinMaxScaler：将属性值缩放到指定的最大和最小值之间
    # Normalizer: 正则化，对每个样本计算其p-范数，再对每个元素除以该范数，这使得每个处理后样本的p-范数（l1-norm,l2-norm）等于1
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer

    # 读取数据
    train_data = pd.read_csv("../dataset/train.csv", header=None)
    train_label_data = pd.read_csv("../dataset/trainLabels.csv", header=None)
    test_data = pd.read_csv("../dataset/test.csv", header=None)

    print(plt.style.available)  # look at available plot styles
    plt.style.use('ggplot')

    X, y = train_data, np.ravel(train_label_data)
    std = StandardScaler()
    X_std = std.fit_transform(X)
    mms = MinMaxScaler()
    X_mms = mms.fit_transform(X)
    norm = Normalizer()
    X_norm = norm.fit_transform(X)
    """
    neig = np.arange(1, 30)
    kfold = 10
    val_accuracy = {'std': [], 'mms': [], 'norm': []}
    best_knn = None
    best_acc = 0.0
    best_scaling = None
    for i, k in enumerate(neig):
        knn = KNeighborsClassifier(n_neighbors=k)
        s1 = np.mean(cross_val_score(knn, X_std, y, cv=kfold))
        val_accuracy['std'].append(s1)
        s2 = np.mean(cross_val_score(knn, X_mms, y, cv=kfold))
        val_accuracy['mms'].append(s2)
        s3 = np.mean(cross_val_score(knn, X_norm, y, cv=kfold))
        val_accuracy['norm'].append(s3)
        if s1 > best_acc:
            best_acc = s1
            best_knn = knn
            best_scaling = 'std'
        elif s2 > best_acc:
            best_acc = s2
            best_knn = knn
            best_scaling = 'mms'
        elif s3 > best_acc:
            best_acc = s3
            best_knn = knn
            best_scaling = 'norm'

    # Plot
    plt.figure(figsize=[13, 8])
    plt.plot(neig, val_accuracy['std'], label='CV Accuracy with std')
    plt.plot(neig, val_accuracy['mms'], label='CV Accuracy with mms')
    plt.plot(neig, val_accuracy['norm'], label='CV Accuracy with norm')
    plt.legend()
    plt.title('k value VS Accuracy')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('Accuracy')
    plt.xticks(neig)
    plt.show()

    print('Best Accuracy with feature scaling:', best_acc)
    print('Best kNN classifier:', best_knn)
    print('Best scaling:', best_scaling)
    best_knn.fit(X_norm, y)

    # 保存文件
    test_fill = np.nan_to_num(test_data)  # 把nan转换为0或inf，-inf
    submission = pd.DataFrame(best_knn.predict(test_fill))
    print(submission.shape)
    submission.columns = ['Solution']
    submission['Id'] = np.arange(1, submission.shape[0] + 1)
    submission = submission[['Id', 'Solution']]
    print(submission.head())

    submission.to_csv('../dataset/submission_with_scaling.csv', index=False)  # index标识是否写列名称
    """
    # 特征选择
    # correlation map
    f, ax = plt.subplots(figsize=(18, 18))
    sns.heatmap(pd.DataFrame(X_std).corr(), annot=True, linewidths=0.5, fmt='.1f', ax=ax)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import f1_score, confusion_matrix
    from sklearn.metrics import accuracy_score

    X_train, X_val, y_train, y_val = train_test_split(X_std, y, test_size=0.3, random_state=42)

    clf_rf = RandomForestClassifier(random_state=43)
    clf_rf = clf_rf.fit(X_train, y_train)

    ac = accuracy_score(y_val, clf_rf.predict(X_val))  # 准确率
    print('Accuracy is: ', ac)
    cm = confusion_matrix(y_val, clf_rf.predict(X_val))  # 混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d')
    plt.show()


if __name__ == "__main__":
    # basic_model()
    model_with_feature_engineering()

