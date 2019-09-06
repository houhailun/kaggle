#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
kaggle入门比赛：Bike Sharing Demand

需求：预测租车数量，因此属于回归类问题

特征：
    'datetime': 日期 2018-08-11 15
    'season': 季节，1 为春季, 2为夏季,3 为秋季,4 为冬季
    'holiday'：当日是否为假期。1代表是，0代表不是
    'workingday'：当日是否为工作日，即既不是周末也不是假期。1代表是，0代表不是。
    'weather'：当日天气： 1: 天气晴朗或者少云/部分有云。 2: 有雾和云/风等。 3: 小雪/小雨，闪电及多云。 4: 大雨/冰雹/闪电和大雾/大雪。
    'temp'：当日摄氏温度
    'atemp'：当日人们感觉的温度
    'humidity'：当日湿度
    'windspeed'：风速
    'casual'：非预定自行车的人数
    'registered': 预定自行车的人数
    'count'：总人数

思路:
一、数据探索
    1、骑车人数必然和季节强相关(春秋多，夏冬少)
    2、骑车人数必然和时候假期、工作日强相关
    3、骑车人数必然和天气强相关
    4、骑车人数必然和当天温度有关
    5、骑车人数必然和风速有关
    6、实际温度和感知温度应该存在某种线性关系
二、特征工程
    1、特征season、weather为类别型变量，可以使用one-hot或者pandas的dummmy方法离散化
    2、变量相关性探索：temp和atemp
    3、datetime可以进一步扩展为年、月、日、周几、几点等
    4、利用RF对特征重要度进行排序，选择重要的特征(但是本项目特征本来就不多，可以不考虑)
    5、gridserchCV对参数进行最优选择
    6、模型融合
    7、模型序列化
"""

import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame, Series
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import ensemble
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


class Solution:
    def load_data(self):
        train_data = pd.read_csv('train.csv')  # [10886 rows x 12 columns]
        test_data = pd.read_csv('test.csv')    #

        return train_data, test_data

    # 数据检查
    def data_analysis(self):
        train_data, _ = self.load_data()
        print(train_data.info(5))  # 查看是否有缺失值，没有缺失值
        print(train_data.columns)  # 查看特征
        print(train_data.head())  # 查看样本
        print(train_data.describe())  # 查看描述信息，均值，最大最小等信息，不存在明显异常点

    # 数据可视化分析
    def data_visualization(self):
        train_data, _ = self.load_data()

        # 1、查看季节于租车总数的关系, 结论：秋季是租车最多，春季租车最小; 但总体相差不多
        # 季节时类别变量，可以住准换为4个01变量
        # season = train_data['season']
        # count = train_data['count']
        # plt.bar(season, count, width=0.1)  # 柱状图
        # plt.xlabel(u"季节")
        # plt.ylabel(u"租车总数")
        # plt.show()

        # 查看每个季节的租车人数
        # print(train_data.groupby(['season'])['count'].sum())  # 春季：312498，夏季：588282，秋季：640662，冬季：544034

        # 2、查看天气与租车的关系, 结论：大雪/大雨天气租车最小
        # weather = train_data['weather']
        # count = train_data['count']
        # plt.bar(weather, count, width=0.1)
        # plt.xlabel(u"天气")
        # plt.ylabel(u"租车总数")
        # plt.show()
        #
        # 查看每个天气对应的租车人数
        # print(train_data.groupby(['weather'])['count'].sum())  # 1:1476063  2: 507160   3:102089    4:164

        # 3、查看温度和租车人数的关系: 20~30度租车人数最多
        # temp = train_data['temp']
        # count = train_data['count']
        # plt.scatter(temp, count)
        # plt.bar(temp, count)
        # plt.xlabel(u'温度')
        # plt.ylabel(u'租车人数')
        # plt.show()
        #
        # print(train_data.groupby(['temp'])['count'].sum())

        # 4、查看是否工作日workingday和租车人数的关系
        # plt.bar(train_data['workingday'], train_data['count'])
        # plt.show()
        #
        # print(train_data.groupby(['workingday'])['count'].sum())  # 0:654872    1:1430604

        # 5、查看当地温度和感知问题的关系, 结论：两者存在某种线性关系，因此只取实际温度即可
        # temp = train_data['temp']
        # atemp = train_data['atemp']
        # plt.scatter(temp, atemp)
        # plt.show()

    # 数据清洗
    def data_clean(self):
        train_data, _ = self.load_data()

        # 1、删除感知温度、未注册人数、注册人数
        train_data.drop(labels='atemp', axis=1, inplace=True)  # 删除atemp列，直接修改
        train_data.drop(['casual'], axis=1, inplace=True)
        train_data.drop(['registered'], axis=1, inplace=True)

        # 2、对类别型变量映射为0/1变量
        train_data["season_label"] = train_data.season.map({1: "Spring", 2: "Summer", 3: "autumn", 4: "Winter"})  # 通过map映射把0，1，2，3转换为春夏秋冬
        season_feat = pd.get_dummies(train_data.season_label)

        train_data["weather_label"] = train_data.season.map({1: "sunny", 2: "cloudy", 3: "rainly", 4: "bad_day"})
        weather_feat = pd.get_dummies(train_data.weather_label)

        train_data = pd.concat([train_data, season_feat, weather_feat], axis=1)          # 3个df结合
        train_data.drop(['season', 'weather', 'season_label', 'weather_label'], axis=1, inplace=True)

        # 3、对datetime进行划分: 年月日，小时，星期几，月份
        train_data['hour'] = train_data['datetime'].str[11:13].astype(int)
        train_data['year'] = train_data['datetime'].str[:4].astype(int)
        train_data['month'] = train_data['datetime'].str[5:7].astype(int)
        train_data['week'] = [datetime.datetime.date(datetime.datetime.strptime(time, '%Y-%m-%d')).weekday() for time in train_data['datetime'].str[:10]]
        train_data.drop(['datetime'], axis=1, inplace=True)

        train_data.to_csv('date_cleaned.csv')

    # 针对清洗后的数据再次深入分析
    def date_analysis(self):
        train_date = pd.read_csv('date_cleaned.csv')

        # 1、热力图查看各变量之间的相关性
        print(train_date.columns)

        corr_mat = train_date[['holiday', 'workingday', 'temp', 'humidity', 'windspeed',
                               'Spring', 'Summer', 'Winter', 'autumn', 'bad_day', 'cloudy',
                               'rainly', 'sunny', 'hour', 'year', 'month', 'week', 'count']].corr()

        # mask = np.array(corr_mat)
        # mask[np.tril_indices_from(mask)] = False
        # plt.figure(figsize=(10, 10))
        # sns.heatmap(corr_mat, mask=mask, square=True, annot=True)
        # plt.show()
        # 结论:节假日和租车人数正相关(弱相关), 工作日和租车人数正相关(弱相关),温度和租车人数正相关(0.39)
        #   湿度是负相关(-0.39), 风度正相关(0.1), 小时正相关(0.4), 年正相关(0.26),月份正相关(0.17)
        #   可以发现：温度、湿度、小时、年、月份比较重要

        # 2、不同月份的租车数
        plt.bar(train_date['month'], train_date['count'], color=['r', 'g', 'b'])
        plt.show()



    def base_model(self):
        # 回归类问题，可以选用线性回归、SVM、决策树、随机森林、GBDT等模型
        train_data, _ = self.load_data()


        # 由于最后评价方法是对数均方
        feature_list = ['holiday', 'workingday', 'temp', 'humidity', 'windspeed', 'Spring', 'Summer', 'Winter',
            'autumn', 'bad_day', 'cloudy', 'rainly', 'sunny', 'hour', 'year', 'month', 'week']

        train_data['log_count'] = np.log(train_data['count'] + 1)

        # 利用GDBT模型
        clf = ensemble.GradientBoostingRegressor()  # 使用默认函数
        clf.fit(train_data[feature_list], train_data['log_count'].values)


if __name__ == "__main__":
    cls = Solution()
    # cls.data_analysis()
    # cls.data_visualization()
    # cls.data_clean()
    cls.date_analysis()

    # cls.base_model()