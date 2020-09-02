#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
描述：本项目属于信用评分的项目，通过改进信用评分技术，预测未来两年借款人会遇到财务困境的可能性
    信用评分算法可以猜测违约概率，这是银行用于确定是否应授予贷款的方法。
    目标是建立一个借款人可以用来帮助做出最佳财务决策的模型。
评估方法：AUC评估

SeriousDlqin2yrs                        150000 non-null int64       超过90天或更糟的逾期拖欠
RevolvingUtilizationOfUnsecuredLines    150000 non-null float64     信用卡和个人信用额度总额,减去不动产和无分期付款债务 除以信用额度总额
age                                     150000 non-null int64       借款人当时的年龄
NumberOfTime30-59DaysPastDueNotWorse    150000 non-null int64       贷款人逾期35-59天的次数，但在过去2年没有更差的信用记录
DebtRatio                               150000 non-null float64     负债比率（每个月债务支出，赡养费和生活费之和 除以 月收入）
MonthlyIncome                           120269 non-null float64     月收入
NumberOfOpenCreditLinesAndLoans         150000 non-null int64       开放式信贷和贷款数量，开放式贷款（分期付款如汽车贷款或抵押贷款）和信贷（如信用卡）的数量
NumberOfTimes90DaysLate                 150000 non-null int64       借款人逾期90天或以上的次数
NumberRealEstateLoansOrLines            150000 non-null int64       抵押贷款和不动产放款的数量，包括房屋净值信贷额度
NumberOfTime60-89DaysPastDueNotWorse    150000 non-null int64       借款人逾期60-89天得到次数, 但在过去2年内没有更糟糕的信用记录
NumberOfDependents                      146076 non-null float64     家属数量：不包括本人在内的家属数量

时间窗口：自变量的观察窗口为过去两年，因变量表现窗口为未来两年。
"""

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns


# windows下反斜杠\是转义字符，想表达\请用\\，或者前面加r
train_file_path = r"D:\code\python\ML\kaggle\Give_Me_Some_Credit\data_set\cs-training.csv"
test_file_path = r"D:\code\python\ML\kaggle\Give_Me_Some_Credit\data_set\cs-test.csv"


class GiveMeSomeCredit:
    def load_data_set(self):
        train_data = pd.read_csv(train_file_path)
        return train_data

    def data_show(self, train_data):
        print('查看info')
        train_data.info()
        # 数据量:15W
        # MonthlyIncome有3W左右条空数据
        # NumberOfDependents有4000左右条空数据

        print('查看数据分布')
        print(train_data.describe())
        print('查看样本head')
        print(train_data.head())

    # 用随机森林对缺失值预测填充函数
    def set_missing(self, df):
        # 把已有的数值型特征取出来,需要把其他的有缺失值的数据去掉否则模型报错
        new_label = ['MonthlyIncome', 'NumberOfDependents']
        new_features = []
        for f in df.columns:
            if f not in new_label:
                new_features.append(f)

        new_label = 'MonthlyIncome'
        # 分成已知该特征和未知该特征两部分
        known_x = df[df.MonthlyIncome.notnull()][new_features]  # 已知实例
        known_y = df[df.MonthlyIncome.notnull()][new_label]     # 已知实例的标签类别
        unknown_x = df[df.MonthlyIncome.isnull()][new_features]  # 未知实例

        # RandomForestRegressor
        rfr = RandomForestRegressor(random_state=0, n_estimators=200, max_depth=3, n_jobs=-1)
        rfr.fit(known_x, known_y)

        # 用得到的模型进行未知特征值预测
        predicted = rfr.predict(unknown_x)

        # 用得到的预测结果填补原缺失数据
        df.loc[(df.MonthlyIncome.isnull()), 'MonthlyIncome'] = predicted
        return df

    def preprocess(self, train_data):
        # 去除id列
        train_data = train_data.ix[:, 1:]  # iloc：通过行号索引行数据;loc：通过行标签索引数据;ix：通过行标签或行号索引数据（基于loc和iloc的混合）

        # step1:缺失值处理:MonthlyIncome有120269个非空数据，NumberOfDependents 有146076个非空数据，缺少比较大
        # 常见的缺失值处理：删除、填充0，填充均值、基于聚类的方法，基于回归的方法
        train_data = self.set_missing(train_data)

        train_data = train_data.dropna()           # 删除比较少的缺失值
        train_data = train_data.drop_duplicates()  # 删除重复项

        # step2:异常值处理
        # 年龄等于0的异常值进行剔除
        train_data = train_data[train_data['age'] > 0]

        # 箱型图
        '''
        data379 = train_data[
            ['NumberOfTime30-59DaysPastDueNotWorse', 'NumberOfTimes90DaysLate', 'NumberOfTime60-89DaysPastDueNotWorse']]
        data379.boxplot()  # 存在异常值96，98
        '''
        # 去除大于90的异常值
        train_data = train_data[train_data['NumberOfTime30-59DaysPastDueNotWorse'] < 90]

        # 拆分X，Y
        train_y = train_data['SeriousDlqin2yrs']
        train_x = train_data.ix[:, 1:]

        X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, test_size=0.3, random_state=0)

        train_data = pd.concat([Y_train, X_train], axis=1)  # 列contact
        test_data = pd.concat([Y_test, X_test], axis=1)
        clasTest = test_data.groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count()  # 查看样本是否均衡
        print(clasTest)
        train_data.to_csv('TrainData.csv', index=False)
        test_data.to_csv('TestData.csv', index=False)

    def feat_analysis(self):
        # 单变量分析，绘制数据分布图，查看数据分布情况
        # 观察好坏客户的整体情况:好客户/坏客户 = 13，样本不均衡,需要均衡样本
        df = pd.read_csv("TrainData.csv")
        grouped = df.groupby('SeriousDlqin2yrs')['SeriousDlqin2yrs'].count()
        print('坏客户占比:%d' %(grouped[0]/grouped[1]))
        # grouped.plot(kind='bar')

        # 单变量分析：age变量数据分布,基本符合正太分布
        age_data = df['age'].groupby(df['age']).count()
        # sns.distplot(age_data, hist=True, kde=True)

        # 箱图可以知道，大部分收入都集中下0~500000，存在个别异常点,暂不处理
        # df[['MonthlyIncome']].boxplot()
        plt.show()


if __name__ == "__main__":
    cls = GiveMeSomeCredit()
    # data = cls.load_data_set()
    # cls.data_show(data)
    # cls.preprocess(data)
    cls.feat_analysis()