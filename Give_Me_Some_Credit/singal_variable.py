#!/usr/bin/env python
# -*- coding:utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame

from imblearn.over_sampling import SMOTE  # 过抽样处理库SMOT

"""
1、样本不平衡处理：SMOTE算法
2、数据离散化：pd.cuts() 或者 卡方分箱
3、woe，iv计算
4、特征值转化为woe
"""


def smote(df):
    print('before smote:')
    df_tmp = df.groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count()  # 对label做分类汇总
    print(df_tmp)

    # 拆分X，Y
    train_y = df.ix[:, :1]
    train_x = df.ix[:, 1:]
    x_cols = train_x.columns
    y_cols = train_y.columns

    # 采用SMOTE算法过采样，平衡化样本
    model_smote = SMOTE()
    x_smote_sample, y_smote_sample = model_smote.fit_sample(train_x, train_y.values.ravel())  # ravel:多维数组转换为1维数组
    x_smote_sample = pd.DataFrame(x_smote_sample, columns=x_cols)
    y_smote_sample = pd.DataFrame(y_smote_sample, columns=y_cols)
    smote_sample = pd.concat([x_smote_sample, y_smote_sample], axis=1)  # 按列合并数据框
    print('after smote:')
    tmp = smote_sample.groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count()  # 对label做分类汇总
    print(tmp)  # 打印输出经过smote处理后的数据集样本分类分布

    return smote_sample


def cut_plot(df):
    """分箱：连续变量离散化"""

    Inf = float('inf')  # 无穷大

    # 离散化age
    bins = (-Inf, 30, 35, 40, 45, 50, 55, 60, 65, 75, Inf)
    df['age_cut'] = pd.cut(df['age'], bins)

    # 离散化 NumberOfTime30 - 59DaysPastDueNotWorse
    bins = (-Inf, 0, 1, 3, 5, 13, Inf)
    df['NumberOfTime30-59DaysPastDueNotWorse_cut'] = pd.cut(df['NumberOfTime30-59DaysPastDueNotWorse'], bins)

    # MonthlyIncome
    bins = (-Inf, 1000, 2000, 3000, 4000, 5000, 6000, 7500, 9500, 12000, Inf)
    df['MonthlyIncome_cut'] = pd.cut(df['MonthlyIncome'], bins)

    # NumberOfTimes90DaysLate
    bins = (-Inf, 0, 1, 3, 5, 10, Inf)
    df['NumberOfTimes90DaysLate_cut'] = pd.cut(df['NumberOfTimes90DaysLate'], bins)

    # NumberRealEstateLoansOrLines
    bins = (-Inf, 0, 1, 2, 3, 5, Inf)
    df['NumberRealEstateLoansOrLines_cut'] = pd.cut(df['NumberRealEstateLoansOrLines'], bins)

    # NumberOfTime60 - 89DaysPastDueNotWorse
    bins = (-Inf, 0, 1, 3, 5, Inf)
    df['NumberOfTime60-89DaysPastDueNotWorse_cut'] = pd.cut(df['NumberOfTime60-89DaysPastDueNotWorse'], bins)

    # NumberOfDependents
    bins = (-Inf, 0, 1, 2, 3, 5, Inf)
    df['NumberOfDependents_cut'] = pd.cut(df['NumberOfDependents'], bins)

    # RevolvingUtilizationOfUnsecuredLines
    bins = (-Inf, 0.25, 0.5, 0.75, 1.0, 2.0, Inf)
    df['RevolvingUtilizationOfUnsecuredLines_cut'] = pd.cut(df['RevolvingUtilizationOfUnsecuredLines'], bins)

    # DebtRatio
    bins = (-Inf, 0.25, 0.5, 0.75, 1.0, 2.0, Inf)
    df['DebtRatio_cut'] = pd.cut(df['DebtRatio'], bins)

    # NumberOfOpenCreditLinesAndLoans
    bins = (-Inf, 5, 10, 15, 20, 25, 30, Inf)
    df['NumberOfOpenCreditLinesAndLoans_cut'] = pd.cut(df['NumberOfOpenCreditLinesAndLoans'], bins)

    return df


def gen_iv(df, var, target):
    """  计算特征的iv值 """

    eps = 0.000001  # 避免除以0

    # 利用交叉表实现特征var对于target的分组频率
    """
    age                 0                1
    (-inf, 30)      6218.000001     8715.000001
    (30, 35)        6551.000001     10737.000001 
    """
    gbi = pd.crosstab(df[var], df[target]) + eps  # 变量每个分组对应target的每个类别的个数
    gb = df[target].value_counts() + eps  # value_counts():统计出target的每个类别的个数
    gbri = gbi / gb
    gbri['woe'] = np.log(gbri[1] / gbri[0])
    gbri['iv'] = (gbri[1] - gbri[0]) * gbri['woe']

    return gbri['iv'].sum()


def gen_woe(df, var, target):
    """  计算woe """
    eps = 0.000001  # 避免除以0

    # 利用交叉表实现特征var对于target的分组频率
    """
    age                 0                1
    (-inf, 30)      6218.000001     8715.000001
    (30, 35)        6551.000001     10737.000001 
    """
    gbi = pd.crosstab(df[var], df[target]) + eps  # 变量每个分组对应target的每个类别的个数
    gb = df[target].value_counts() + eps  # value_counts():统计出target的每个类别的个数
    gbri = gbi / gb
    gbri['woe'] = np.log(gbri[1] / gbri[0])

    return gbri['woe'].to_dict()


def woe_iv(df):
    """ 计算WOE，IV，用于后续特征选择 """

    target = 'SeriousDlqin2yrs'

    # 离散化后的变量
    col_list = []
    for col in df.columns[1:]:
        if col.endswith('_cut'):
            col_list.append(col)

    col_iv_map = {}
    col_woe_map = {}
    # 对每个特征的每个分组计算WOEi和IVi
    for vol in col_list:
        iv = gen_iv(df, vol, target)
        woe_map = gen_woe(df, vol, target)
        col_iv_map[vol] = iv
        col_woe_map[vol] = woe_map

    return col_woe_map, col_iv_map


def value_2_woe(df, woe_map):
    """  把变量的值替换为woe """
    target = 'SeriousDlqin2yrs'
    col_list = []
    for col in df.columns[1:]:
        if not col.endswith('_cut') and col != target:
            col_list.append(col)

    # 双重循环，替换col变量为woe
    for col in col_list:
        for i in range(len(df[col])):
            val = df[col][i]
            for cut, woe in woe_map[col+'_cut'].items():
                if val in cut:
                    df[col][i] = woe

    # 删除分箱变量
    for col in df.columns:
        if col.endswith('_cut'):
            df.drop([col], axis=1, inplace=True)
    print(df.head(5))
    return df


if __name__ == "__main__":
    df = pd.read_csv('TrainData.csv')

    df = smote(df)  # smote平衡

    df = cut_plot(df)  # 分箱

    # woe_map, iv_map = woe_iv(df)  # iv值df
    # print(woe_map)
    #
    # # FIXME:特征值转换为采用两次for循环遍历导致效率及其慢
    # df = value_2_woe(df, woe_map)  # 把特征值转换未woe
    #
    # df.to_csv('woe_train.csv', index=False)

    # LR模型
