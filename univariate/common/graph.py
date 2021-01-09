import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(palette='muted', color_codes=True)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# plot density and hist graph
def density_hist_curve(x, xlab='x', ylab='y', density=True):
    # x : data for plot density curve
    # density : True or False, True means there is a density plot in graph
    # sns.set(palette="muted", color_codes=True)
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    figure = sns.distplot(x, kde=density, kde_kws={"shade": True})
    figure.set_xlabel(xlab)
    figure.set_ylabel(ylab)
    return figure


# boxplot
def boxplot(y1, x1=None, z1=None, xlab='x', ylab='y'):
    # x1: xlabel data, string type
    # y1: ylabel data
    # z1: group
    # xlab: the name of x
    # sns.set(palette='muted', color_codes=True)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    figure = sns.boxplot(x=x1, y=y1, hue=z1)
    figure.set_xlabel(xlab)
    figure.set_ylabel(ylab)
    return figure


# scatter plot
def scatter_continuous(x1, y1, xlab='x', ylab='y', title1='scatter plot'):
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.title(title1)
    figure = plt.scatter(x1, y1)
    return figure


# scatter plot with categorical data
def scatter_category(x1, y1, xlab='x', ylab='y'):
    # x1: categorical data
    # y1: continuous data
    # sns.set(palette='muted', color_codes=True)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    figure = sns.stripplot(x=x1, y=y1)
    figure.set_xlabel(xlab)
    figure.set_ylabel(ylab)
    return figure


# line chart
def line_chart(x1, y1, z1=None, xlab='x', ylab='y', title1='line chart', rotation1=45):
    # x1:x轴数据
    # y1:y轴数据
    # z1:分组数据
    # rotation: x轴标签旋转角度
    # sns.set(palette='muted', color_codes=True)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(8, 6.5))  # 设置图像大小
    plt.xticks(rotation=rotation1)
    plt.title(title1)
    figure = sns.pointplot(x=x1, y=y1, hue=z1, scale=0.75)
    figure.set_xlabel(xlab)
    figure.set_ylabel(ylab)
    return figure


# density graph
# 分别绘制指标在几种取值下的密度曲线图
def density_graph(x1, y1, xlab='xxx', ylab='yyy', group='zzz', shade1=True):
    # x1: 第一类数据
    # y1: 第二类数据
    # xlab: x1对应的name
    # ylab: y1对应的name
    # group: 分类name
    # sns.set(palette='muted', color_codes=True)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    df1 = pd.DataFrame({xlab: x1, group: y1})
    figure = sns.FacetGrid(df1, hue=group)
    figure = figure.map(sns.kdeplot, xlab, shade=shade1)
    plt.legend()
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    return figure


# 分组柱状图
def barplot_group(x1, y1, z1=None, xlab='x', ylab='y'):
    # sns.set(palette='muted', color_codes=True)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    if all(z1) == None:
        figure = sns.barplot(x1, y1, ci=0)
    else:
        figure = sns.barplot(x1, y1, hue=z1, ci=0)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    return figure


# 对因子变量计数，然后绘制条形图
def count_plot(x1, y1=None, xlab='x', ylab='y'):
    # x1: 分类数据
    # y1: 分组数据
    # sns.set(palette='muted', color_codes=True)
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # plt.rcParams['axes.unicode_minus'] = False
    if all(x1) == None:
        figure = sns.countplot(x1)
    else:
        figure = sns.countplot(x1, hue=y1)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    return figure
