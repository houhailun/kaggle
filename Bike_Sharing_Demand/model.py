#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
kaggle入门比赛：Bike Sharing Demand

需求：预测租车数量，因此属于回归问题

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

import pickle
import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn import ensemble

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score


class Solution:
    # 预测租车数量方案
    def __init__(self):
        self.fp_train = 'train.csv'
        self.fp_test = 'test.csv'

    def load_data(self):
        # 加载训练集和测试集
        train_data = pd.read_csv(self.fp_train)  # [10886 rows x 12 columns]
        test_data = pd.read_csv(self.fp_test)    #

        return train_data, test_data

    def data_analysis(self):
        # 数据EDA检查
        # 查看样本分布，是否有缺失，有无明显异常值，有无重复值，有无极限值等
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
        df = train_data.groupby(['season'], as_index=False)['count'].sum()  # 春季：312498，夏季：588282，秋季：640662，冬季：544034
        print(df)
        sns.barplot(x='season', y='count', data=df)
        plt.xlabel(u"season")
        plt.ylabel(u"bike count")
        plt.title(u'season bike count')
        plt.show()

        # 2、查看天气与租车的关系, 结论：大雪/大雨天气租车最小
        df = train_data.groupby(['weather'], as_index=False)['count'].sum()  # 1:1476063  2: 507160   3:102089    4:164
        print(df)
        sns.barplot(x='weather', y='count', data=df)
        plt.xlabel(u"weather")
        plt.ylabel(u"bike count")
        plt.title(u'weather bike count')
        plt.show()

        # 3、查看温度和租车人数的关系: 20~30度租车人数最多
        df = train_data.groupby(['temp'], as_index=False)['count'].sum()
        print(df)
        sns.scatterplot(x='temp', y='count', data=df)
        plt.xlabel(u'temp')
        plt.ylabel(u'count')
        plt.show()

        # 4、查看是否工作日workingday和租车人数的关系
        df = train_data.groupby(['workingday'], as_index=False)['count'].sum()
        print(df)
        sns.barplot(x='workingday', y='count', data=df)
        plt.xlabel(u'workingday')
        plt.ylabel(u'bike count')
        plt.show()

        # 5、查看当地温度和感知问题的关系, 结论：两者存在某种线性关系，因此只取实际温度即可
        sns.jointplot(x='temp', y='atemp', data=train_data, kind='reg')
        plt.xlabel('temp')
        plt.ylabel('atemp')
        plt.title('temp vs atemp')
        plt.show()

    def feature_engineering(self, mode='train'):
        # 特征工程
        print('feature_engineering()')

        train_data, test_data = self.load_data()

        data = train_data
        if mode == 'test':
            data = test_data

        print(data.columns)
        data.drop(['atemp'], axis=1, inplace=True)

        if mode == 'train':
            data.drop(['casual'], axis=1, inplace=True)
            data.drop(['registered'], axis=1, inplace=True)

        # 2、对类别型变量映射为0/1变量
        data["season_label"] = data.season.map({1: "Spring", 2: "Summer", 3: "autumn", 4: "Winter"})  # 通过map映射把0，1，2，3转换为春夏秋冬
        season_feat = pd.get_dummies(data.season_label)
        print(season_feat.head(5))

        data["weather_label"] = data.weather.map({1: "sunny", 2: "cloudy", 3: "rainly", 4: "bad_day"})
        weather_feat = pd.get_dummies(data.weather_label)
        print(weather_feat.head(5))

        data = pd.concat([data, season_feat, weather_feat], axis=1)          # 3个df结合
        data.drop(['season', 'weather', 'season_label', 'weather_label'], axis=1, inplace=True)

        # 3、对datetime进行划分: 年月日，小时，星期几，月份
        data['hour'] = data['datetime'].str[11:13].astype(int)
        data['year'] = data['datetime'].str[:4].astype(int)
        data['month'] = data['datetime'].str[5:7].astype(int)
        data['week'] = [datetime.datetime.date(datetime.datetime.strptime(time, '%Y-%m-%d')).weekday()+1 for time in data['datetime'].str[:10]]
        # data.drop(['datetime'], axis=1, inplace=True)

        file_name = 'data_cleaned'
        if mode == 'test':
            file_name = file_name + '_test'
        data.to_csv(file_name+'.csv', index=False)

    def feature_engineering_analysis(self):
        # 针对处理后的数据再次深入分析
        train_data = pd.read_csv('date_cleaned.csv')

        # 1、热力图查看各变量之间的相关性
        print(train_data.columns)

        corr_mat = train_data[['holiday', 'workingday', 'temp', 'humidity', 'windspeed',
                               'Spring', 'Summer', 'Winter', 'autumn', 'bad_day', 'cloudy',
                               'rainly', 'sunny', 'hour', 'year', 'month', 'week', 'count']].corr()

        mask = np.array(corr_mat)
        mask[np.tril_indices_from(mask)] = False
        plt.figure(figsize=(10, 10))
        sns.heatmap(corr_mat, mask=mask, square=True, annot=True)
        plt.show()
        # 结论: 温度、小时、年、月份比较重要；湿度呈现较大的负相关，因为天气越潮湿，骑车人越少

        # 2、租车辆按月份、年份统计
        fig, axes = plt.subplots(3, 1, figsize=(12, 20))
        month_Aggregated = pd.DataFrame(train_data.groupby(['year', 'month'])['count'].mean()).reset_index()
        sns.barplot(data=month_Aggregated, x="month", y="count", hue='year', ax=axes[0])
        axes[0].set(xlabel='Month', ylabel='Avearage Count', title="Average Count By Month")

        # # 每日租车辆按季节统计
        hour_Aggregated1 = pd.DataFrame(train_data.groupby(['hour'])['count'].mean()).reset_index()
        sns.pointplot(data=hour_Aggregated1, x='hour', y='count', join=True, ax=axes[1])
        axes[1].set(xlabel='Hour', ylabel='Avearage Count', title="Average Count By Hour Across Season")

        # 日租车辆按周统计
        hour_Aggregated2 = pd.DataFrame(train_data.groupby(['hour', 'week'])['count'].mean()).reset_index()
        sns.pointplot(data=hour_Aggregated2, x='hour', y='count', hue='week', join=True, ax=axes[2])
        axes[2].set(xlabel='Hour', ylabel='Avearage Count', title="Average Count By Hour Across weekday")
        plt.show()
        # 小结：
        # 图1可以看出2012年共享单车的使用量高于2011年，消费人群增加了1.5~2倍。两年内租车量随月份变化的趋势相同，6、7、8月有明显的高需求。
        # 图2可以看出租车时间高峰为上午7-8点，下午5-6点，符合上下班通勤的时间范围。
        # 图3可以看出工作日租车辆主要为上下班时间，周末租车辆主要集中在10am-4pm之间。

    def base_model(self):
        # 回归类问题，可以选用线性回归、SVM、决策树、随机森林、GBDT等模型
        train_data = pd.read_csv('data_cleaned.csv')

        feature_list = ['holiday', 'workingday', 'temp', 'humidity', 'windspeed', 'Spring', 'Summer', 'Winter',
                        'autumn', 'bad_day', 'cloudy', 'rainly', 'sunny', 'hour', 'year', 'month', 'week']

        train_data['log_count'] = np.log(train_data['count'] + 1)  # 由于最后评价方法是对数均方

        print(train_data.columns)

        # 划分训练集，验证集比例为8：2
        X = train_data[feature_list]
        y = train_data['log_count']
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
        print(X_train.shape)  # (8708, 17)
        print(X_test.shape)   # (2178, 17)

        # # 利用 GDBT 模型
        # clf = ensemble.GradientBoostingRegressor()  # 使用默认函数
        # clf.fit(X_train, y_train)
        # print("Training score:%f" % clf.score(X_train, y_train))  # Training score:0.926539
        # print('模型重要度：', clf.feature_importances_)
        #
        # y_pred = clf.predict(X_test)
        # print('mean_squared_log_error score:%f' % mean_squared_log_error(y_test, y_pred))  # Test score:0.010021
        # print('r2_score score:%f' % r2_score(y_test, y_pred))

        """
        参数说明:
        影响树的参数：
            min_samples_split: 用于定义一个节点可分裂的最少的样本数量；控制过拟合，值越小越容易过拟合，值太大会容易欠拟合;
                通常在样本量的0.5-1%之间，但如果是不平衡样本，值可以在设置的低一些
            min_samples_leaf: 叶子节点最少的样本数量; 控制过拟合; 平衡样本需要设置较小的值，因为少数样本为一类的叶子样本量会偏小
            max_depth: 树的深度；控制过拟合
            max_leaf_nodes：最终的数的叶子节点的最多数量；与max_depth能相互替代，如果被定义了，max_depth将被忽视
            max_features：当寻找最优分裂点时，考虑的随机选择的特征的数目；经验原则：特征数目的平方根很好用，但是这个数目应该有总特征数目的30%-40%以上
        影响模型的参数：
            learning_rate：
            n_estimators：建模的树的数量；虽然GBM对于大额n_estimators值具有较强的鲁棒性但是在某些点的时候依然会产生过拟合问题，因此需要通过CV在特定的learning_rate下进行调参
            subsample：每棵树的样本数量，随机选择；值小于1能通过降低模型的方差增强模型的鲁棒性；常用的值范围[0,0.8]，也可以调高点
        求它参数：
            loss: 节点分裂时的损失函数
            random_state：随机种子，固定随机种子后，模型里面的随机化过程将会被重复；
                这个参数对于模型调优非常重要，如果不固定，将很难进行模型调优，无法对比模型结果的好坏
        """

        # 模型调参1：调整决策树个数
        def param1():
            param_test1 = {'n_estimators': range(200, 301, 10)}
            gsearch1 = GridSearchCV(estimator=ensemble.GradientBoostingRegressor(learning_rate=0.1,
                                                                                 max_depth=8,
                                                                                 max_features='sqrt',
                                                                                 subsample=0.8,
                                                                                 random_state=42,
                                                                                 loss='ls'),
                                    param_grid=param_test1,
                                    cv=5)
            gsearch1.fit(X_train, y_train)
            print('best params:', gsearch1.best_params_)  # {'n_estimators': 250}
            print('best score:', gsearch1.best_score_)  # 0.9499609451542114

            y_pred = gsearch1.predict(X_test)
            print('test mean_squared_log_error score:%f' % mean_squared_log_error(y_test, y_pred))  # 0.007108
            print('test r2_score score:%f' % r2_score(y_test, y_pred))  # 0.954516

        # param1()

        def param2():
            # 对决策树进行调参,首先我们对决策树最大深度max_depth和内部节点再划分所需最小样本数min_samples_split进行网格搜索
            param_test2 = {'max_depth': range(3, 14, 2), 'min_samples_split': range(5, 101, 20)}
            gsearch1 = GridSearchCV(estimator=ensemble.GradientBoostingRegressor(n_estimators=250,
                                                                                 learning_rate=0.1,
                                                                                 max_depth=8,
                                                                                 max_features='sqrt',
                                                                                 subsample=0.8,
                                                                                 random_state=42,
                                                                                 loss='ls'),
                                    param_grid=param_test2,
                                    cv=5)
            gsearch1.fit(X_train, y_train)
            print('best params:', gsearch1.best_params_)  # {'max_depth': 7, 'min_samples_split': 85}
            print('best score:', gsearch1.best_score_)  # 0.9548697645529288

            y_pred = gsearch1.predict(X_test)
            print('test mean_squared_log_error score:%f' % mean_squared_log_error(y_test, y_pred))  # 0.006693
            print('test r2_score score:%f' % r2_score(y_test, y_pred))  # 0.955979

        # param2()

        # 由于深度7是合理值，因此固定下，再对内部节点再划分所需最小样本数min_samples_split和叶子节点最少样本数min_samples_leaf一起调参。
        def param3():
            param_test3 = {'min_samples_split': range(50, 100, 10), 'min_samples_leaf': range(10, 101, 10)}
            gsearch1 = GridSearchCV(estimator=ensemble.GradientBoostingRegressor(n_estimators=250,
                                                                                 learning_rate=0.1,
                                                                                 max_depth=7,
                                                                                 max_features='sqrt',
                                                                                 subsample=0.8,
                                                                                 random_state=42,
                                                                                 loss='ls'),
                                    param_grid=param_test3,
                                    cv=5)
            gsearch1.fit(X_train, y_train)
            print('best params:', gsearch1.best_params_)  # {'min_samples_leaf': 10, 'min_samples_split': 60}
            print('best score:', gsearch1.best_score_)  # 0.9556712088916134

            y_pred = gsearch1.predict(X_test)
            print('test mean_squared_log_error score:%f' % mean_squared_log_error(y_test, y_pred))  # 0.006316
            print('test r2_score score:%f' % r2_score(y_test, y_pred))  # 0.957869

        # param3()

        def param4():
            # 最大特征进行调参
            param_test4 = {'max_features': range(4, 18, 2)}
            gsearch4 = GridSearchCV(
                estimator = ensemble.GradientBoostingRegressor(learning_rate=0.1, n_estimators=250, max_depth=7,
                                                               min_samples_leaf=10,
                                                               min_samples_split=60, subsample=0.8, random_state=42),
                param_grid=param_test4, cv=5)
            gsearch4.fit(X_train, y_train)

            print('best params:', gsearch4.best_params_)  # {'max_features': 16}
            print('best score:', gsearch4.best_score_)  # 0.9597117682242261

            y_pred = gsearch4.predict(X_test)
            print('test mean_squared_log_error score:%f' % mean_squared_log_error(y_test, y_pred))  # 0.005964
            print('test r2_score score:%f' % r2_score(y_test, y_pred))  # 0.961180

        # param4()

        def param5():
            # 子采样的比例进行网格搜索
            param_test5 = {'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]}
            gsearch4 = GridSearchCV(
                estimator = ensemble.GradientBoostingRegressor(learning_rate=0.1, n_estimators=250, max_depth=7,
                                                               min_samples_leaf=10, max_features=16,
                                                               min_samples_split=60, random_state=42),
                param_grid=param_test5, cv=5)
            gsearch4.fit(X_train, y_train)

            print('best params:', gsearch4.best_params_)  # {'subsample': 0.8}
            print('best score:', gsearch4.best_score_)  # 0.9597117682242261

            y_pred = gsearch4.predict(X_test)
            print('test mean_squared_log_error score:%f' % mean_squared_log_error(y_test, y_pred))  # 0.005964
            print('test r2_score score:%f' % r2_score(y_test, y_pred))  # 0.961180

        # param5()

        clf = ensemble.GradientBoostingRegressor(learning_rate=0.05, n_estimators=250, max_depth=7,
                                                 subsample=0.8, min_samples_leaf=10, max_features=16,
                                                 min_samples_split=60, random_state=42)
        clf.fit(X, y)
        print(X.columns)
        print('模型重要度：', clf.feature_importances_)


        # y_pred = clf.predict(X_test)
        # print('test mean_squared_log_error score:%f' % mean_squared_log_error(y_test, y_pred))  # 0.005952
        # print('test r2_score score:%f' % r2_score(y_test, y_pred))  # 0.960812

        # 模型持久化
        with open('gbdt_param.pickle', 'wb') as fw:
            pickle.dump(clf, fw)

        # # 加载模型
        # with open('gbdt_param.pickle', 'rb') as fr:
        #     clf = pickle.load(fr)
        #
        # y_pred = clf.predict(X_test)
        # print('mean_squared_log_error score:%f' % mean_squared_log_error(y_test, y_pred))  # Test score:0.010021
        # print('r2_score score:%f' % r2_score(y_test, y_pred))                               # Test score: 0.922330

    def predict(self):
        # 预测

        # 对测试数据做同样预处理
        self.feature_engineering(mode='test')

        # 加载模型预测
        with open('gbdt_param.pickle', 'rb') as fr:
            clf = pickle.load(fr)

        test_data = pd.read_csv('data_cleaned_test.csv')

        y_pred = clf.predict(test_data.drop('datetime', axis=1))

        y_pred = pd.DataFrame({'datetime': test_data['datetime'],
                               'count': np.exp(y_pred)-1})

        y_pred = y_pred[['datetime', 'count']]
        y_pred.to_csv('submit.csv', index=False)


if __name__ == "__main__":
    cls = Solution()

    cls.data_analysis()
    cls.data_visualization()

    cls.feature_engineering()
    cls.feature_engineering_analysis()

    cls.base_model()

    cls.predict()
