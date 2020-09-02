#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/8/12 10:23
# Author: Hou hailun

# 模型预处理

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute._iterative import IterativeImputer


# windows下反斜杠\是转义字符，想表达\请用\\，或者前面加r
train_file_path = r"D:\code\python\ML\kaggle\Give_Me_Some_Credit\data_set\cs-training.csv"
test_file_path = r"D:\code\python\ML\kaggle\Give_Me_Some_Credit\data_set\cs-test.csv"

train = pd.read_csv(train_file_path)
# test = pd.read_csv(test_file_path)
#
# y = train['SeriousDlqin2yrs']
# train = train.drop(['SeriousDlqin2yrs', 'Unnamed: 0'], axis=1)
# test = test.drop(['SeriousDlqin2yrs', 'Unnamed: 0'], axis=1)
#
# from lightgbm import LGBMClassifier
# lgb = LGBMClassifier(colsample_bytree=0.5,
#                      subsample=0.8,
#                      num_leaves=20,
#                      n_estimators=1200,
#                      learning_rate=0.0075)
# lgb.fit(train, y)
#
# result = lgb.predict_proba(test)
# print(pd.DataFrame(lgb.feature_importances_))
#
# # 查看特征重要性
# importance_df = pd.DataFrame(lgb.feature_importances_).rename(columns={0: "importance"})
# importance_df['columns'] = train.columns
# importance_df['importance'] = (importance_df['importance'] / importance_df['importance'].values.sum())*100
# importance_df = importance_df.sort_values("importance", ascending=False)
# importance_df.head(10)
#
#
# sample=pd.read_csv("/kaggle/input/GiveMeSomeCredit/sampleEntry.csv")
# sample.head()
# sample["Probability"]=result[:,1]
# sample.head()
# sample.to_csv("20191007_LGBMClassifier.csv", index=False)


# 数据EDA
# 样本是否平衡
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
sns.countplot(x='SeriousDlqin2yrs', data=train, ax=axes[0][0])
print("proportion of people who defaulted: {}".format(train['SeriousDlqin2yrs'].sum() / len(train)))
# proportion of people who defaulted: 0.06684

# 检查特征的缺失值个数及所占比例
null_val_sums = train.isnull().sum()
print(pd.DataFrame({"Column": null_val_sums.index,
                    "Num of null valueL": null_val_sums.values,
                    "Proportion": null_val_sums.values / len(train)}))
# MonthlyIncome               29731    0.198207
#  NumberOfDependents         3924     0.026160

# 信用总额
# print(train['RevolvingUtilizationOfUnsecuredLines'].describe())
# sns.distplot(train['RevolvingUtilizationOfUnsecuredLines'], ax=axes[0][1])

# 查看age
print(train['age'].describe())
sns.distplot(train['age'], ax=axes[0][1])  # 基本符合正太分布，有个别小于0需要剔除
sns.distplot(train.loc[train['SeriousDlqin2yrs'] == 0]['age'], ax=axes[1][0])
sns.distplot(train.loc[train['SeriousDlqin2yrs'] == 1]['age'], ax=axes[1][1])

plt.show()