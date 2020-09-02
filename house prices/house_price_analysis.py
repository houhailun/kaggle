#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2019/9/23 14:08
# Author: Hou hailun

# Index(['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',
        #        'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
        #        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
        #        'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
        #        'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
        #        'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
        #        'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',
        #        'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating',
        #        'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF',
        #        'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',
        #        'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',
        #        'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',
        #        'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',
        #        'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',
        #        'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',
        #        'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',
        #        'SaleCondition', 'SalePrice']

# 分析影响房价的可能因素，并预测房价，属于回归类问题
# 读数据，看分布，查关联，找异常，填空值，转非数，做验证，做预测，交结果

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

plt.style.use('ggplot')  # 图像美化


class HousePrice:

    def __init__(self):
        self.train_data = pd.read_csv('datasets/train.csv')

    def data_explore(self):
        # 数据探索
        print(' train.shape '.center(40, '*'))
        print(self.train_data.shape)  # (1460, 81)

        print(' train.info() '.center(40, '*'))
        # self.train_data.info()  # 部分数据存在空数据

        print(' train.describe() '.center(40, '*'))
        # print(self.train_data.describe())

        print(' train.columns '.center(40, '*'))
        # print(self.train_data.columns)

        print("train['SalePrice'].describe()".center(40, '*'))
        print(self.train_data['SalePrice'].describe())
        # 小结：价格都大于0，无明显异常

    def data_visualization(self):
        # 数据可视化，各个数值属性与Y值的关联程度，去除关联程度非常低的属性
        data = self.train_data.corr()
        # sns.heatmap(data)
        print(data['SalePrice'].sort_values())
        # 越是白色越是关联紧密。可以观察SalePrice跟哪些属性关联更紧密
        # 热力图显示有很多时候显示不全，尤其属性特别多的时候。这个时候，可能直接查看更方便。
        # 可以看到有很多特征和因变量的相关系数是负值，可以直接把低于0.2的特征删除

        # print(' SalePrice show '.center(40, '*'))
        # 集合了matplotlib的hist()与核函数估计kdeplot的功能，增加了rugplot分布观测条显示与利用scipy库fit拟合参数分布的新颖用途
        # sns.distplot(self.train_data['SalePrice'])
        # print("skewness: %f" % self.train_data['SalePrice'].skew())
        # print("Kurtosis: %f" % self.train_data['SalePrice'].kurt())
        # 小结: 房屋售价符合正态分布；有明显的正偏度
        # 偏度(skewness)和峰度(kurtosis）  skewness就是三阶中心距，kurotosis就是4阶的
        # sk表示的是左偏还是右偏（ > 0, 右偏）（< 0左偏），=0就是正态分布
        # ku是尾巴的胖瘦或者高低。峰度包括正态分布（峰度值 = 3），厚尾（峰度值 > 3），瘦尾（峰度值 < 3）
        # 一般希望特征分布最好符合normal distribution。是否符合则是检测数据的skewness。如果skew()
        # 值大于0.75， 则需要进行分布变换。一般使用log变换

        print(' 土地面积和售价的关系 '.center(40, '*'))
        # var = 'GrLivArea'
        # data = pd.concat([self.train_data['SalePrice'], self.train_data[var]], axis=1)
        # data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
        # 小结: 可以看到售价整体和面积成线性关系，但是有2个异常点(面积很大，但是售价低)，需要剔除掉
        # self.train_data = self.train_data[-((self.train_data.SalePrice < 200000) &
        # (self.train_data.GrLivArea > 4000))]

        print(' TotalBsmtSF-SalePrice '.center(40, '*'))
        # var = 'TotalBsmtSF'
        # data = pd.concat([self.train_data['SalePrice'], self.train_data[var]], axis=1)
        # data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
        # 小结：可以看到TotalBsmtSF和售价存在线性关系，但是有1个异常点
        # self.train_data = self.train_data[-((self.train_data['TotalBsmtSF'] > 6000) & (self.train_data['SalePrice'] < 200000))]

        print(' 类型变量OverallQual和售价的关系 '.center(40, '*'))
        # var = 'OverallQual'
        # data = pd.concat([self.train_data['SalePrice'], self.train_data['OverallQual']], axis=1)
        # f, ax = plt.subplots(figsize=(8, 6))
        # fig = sns.boxplot(x=var, y='SalePrice', data=data)
        # fig.axis(ymin=0, ymax=800000)
        # 可以看到两者分布趋势相同

        print('类型变量YearBuilt和售价的关系'.center(40, '*'))
        # var = 'YearBuilt'
        # data = pd.concat([self.train_data['SalePrice'], self.train_data[var]], axis=1)
        # f, ax = plt.subplots(figsize=(8, 6))
        # fig = sns.boxplot(x=var, y='SalePrice', data=data)
        # fig.axis(ymin=0, ymax=800000)
        # 小结：两个变量之间没有很强的趋势性，但可以看到建筑时间短的房屋价格更高
        # 总结：'GrLivArea'和'TotalBsmtSF'与'SalePrice'似乎线性相关，并且都是正相关。 对于'TotalBsmtSF'，线性关系的斜率十分的高。·'OverallQual'和
        # 'YearBuilt'与'SalePrice'也有关系。'OverallQual'的相关性更强, 箱型图显示了随着整体质量的增长，房价的增长趋势

        plt.show()

    def data_corr(self):
        # 相关系数分析
        # 热力图: 观察那些变量和预测目标关系大，那些变量之间会有较强的关联
        corrmat = self.train_data.corr()
        # f, ax = plt.subplots(figsize=(12, 9))
        # sns.heatmap(corrmat, vmax=0.8, square=True)
        # 首先两个红色的方块吸引到了我，第一个是'TotalBsmtSF' 和'1stFlrSF' 变量的相关系数，第二个是 'GarageX' 变量群。这两个示例都显示了这些变量之间很强的相关性。实际上，相关性的程度达到了一种多重共线性的情况。我们可以总结出这些变量几乎包含相同的信息，所以确实出现了多重共线性。
        # 另一个引起注意的地方是 'SalePrice' 的相关性。我们可以看到我们之前分析的 'GrLivArea','TotalBsmtSF',和 'OverallQual'的相关性很强，除此之外也有很多其他的变量应该进行考虑，

        # 选择
        print('saleprice相关系数矩阵，特征选择')
        # k = 10
        # cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index  # 选择和saleprice系数最大的10个变量
        # print(cols)
        # cm = np.corrcoef(self.train_data[cols].values.T)
        # sns.set(font_scale=1.25)
        # hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10},
        #                  yticklabels = cols.values, xticklabels=cols.values)
        # 1、'OverallQual', 'GrLivArea'以及'TotalBsmtSF'与'SalePrice'有很强的相关性
        # 2、'GarageCars'和'GarageArea'也是相关性比较强的变量.车库中存储的车的数量是由车库的面积决定的，它们就像双胞胎，所以不需要专门区分'GarageCars'和'GarageArea' ，所以我们只需要其中的一个变量。这里我们选择了'GarageCars'因为它与'SalePrice'的相关性更高一些
        # 3、'TotalBsmtSF'和'1stFloor'与上述情况相同，我们选择'TotalBsmtSF' 。
        # 4、'FullBath'几乎不需要考虑。
        # 5、'TotRmsAbvGrd'和'GrLivArea'也是变量中的双胞胎。'YearBuilt'和'SalePrice'相关性似乎不强。

        # 绘制图形查看挑选的变量与售价的关系
        sns.set()
        cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']
        sns.pairplot(self.train_data[cols], size=2.5)
        plt.show()

    def data_missing(self):
        # 缺失值处理
        # isnull().sum(): 把列中为空的个数统计出来
        # isnull().count():
        total = self.train_data.isnull().sum().sort_values(ascending=False)  # 对缺失个数排序
        percent = (self.train_data.isnull().sum() / self.train_data.isnull().count()).sort_values(ascending=False)
        missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
        print(missing_data.head(20))
        # 当超过15 % 的数据都缺失的时候，我们应该删掉相关变量且假设该变量并不存在。
        # 根据这一条，一系列变量都应该删掉，例如'PoolQC', 'MiscFeature', 'Alley'等等，这些变量都不是很重要，因为他们基本都不是我们买房子时会考虑的因素。
        # 'GarageX'变量群的缺失数据量都相同，由于关于车库的最重要的信息都可以由'GarageCars'表达，并且这些数据只占缺失数据的5 %，我们也会删除上述的'GarageX'变量群。
        # 同样的逻辑也适用于'BsmtX'变量群。
        # 对于'MasVnrArea'和'MasVnrType', 我们可以认为这些因素并不重要。
        # 除此之外，他们和'YearBuilt''OverallQual'都有很强的关联性，而这两个变量我们已经考虑过了。所以删除'MasVnrArea'和'MasVnrType'并不会丢失信息。
        # 最后, 由于'Electrical'中只有一个缺失的观察值，所以我们删除这个观察值，但是保留这一变量。
        # 此外, 我们还可以通过补充缺失值, 通过实际变量的含义进行补充, 例如类别型变量, 就可以补充成No, 数值型变量可以补充成0, 或者用平均值来填充。
        self.train_data = self.train_data.drop((missing_data[missing_data['Total'] > 1]).index, 1)
        self.train_data = self.train_data.drop(self.train_data.loc[self.train_data['Electrical'].isnull()].index)
        print(self.train_data.isnull().sum().max())  # just checking that there's no missing data missing...

    def outlier_process(self):
        # 异常点处理
        # 标准化：均值为0，方差为1
        saleprice_scaled = StandardScaler().fit_transform(self.train_data['SalePrice'][:, np.newaxis]);
        # argsort: 返回的是数组值从小到大的索引值
        low_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][:10]
        high_range = saleprice_scaled[saleprice_scaled[:, 0].argsort()][-10:]
        print('outer range(low) if the distribution:')
        print(low_range)
        print('outer range(high) if the distribution:')
        print(high_range)

        # 两变量分析
        # saleprice/grlivarea
        # var = 'GrLivArea'
        # data = pd.concat([self.train_data['SalePrice'], self.train_data[var]], axis=1)
        # data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
        # # 小结：可以看到有2个异常点，面积很大，价格却很低，不符合整体线性关系
        # self.train_data = self.train_data[-((self.train_data.SalePrice < 200000) &
        #                                     (self.train_data.GrLivArea > 4000))]

        # saleprice/TotalBsmtSF
        var = 'TotalBsmtSF'
        data = pd.concat([self.train_data['SalePrice'], self.train_data[var]], axis=1)
        data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
        # 可以看到有1个异常点，TotalBsmtSF很大，价格却很低，不符合整体线性关系
        #         # self.train_data = self.train_data[-((self.train_data.SalePrice < 200000) &
        #         #                                     (self.train_data.GrLivArea > 6000))]

        plt.show()



if __name__ == "__main__":
    house_price = HousePrice()
    # house_price.data_explore()
    # house_price.data_visualization()
    # house_price.data_corr()
    # house_price.data_missing()
    house_price.outlier_process()