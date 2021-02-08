# -*- coding: utf-8 -*-
# Group: MI
# Author: zhao chen
# Date: 2020-07-16


import toad
import pandas as pd
import numpy as np
import pydotplus
from IPython.display import Image
from sklearn.externals.six import StringIO
import os
from sklearn import tree
import graphviz


'''
使用决策树，输出特征分裂图
'''

class auto_tree(object):

    def __init__(self, datasets, ex_lis, dep='bad_ind', min_samples=0.05, min_samples_leaf=200, min_samples_split=20,
                 max_depth=4,  is_bin=True):

        '''
        datasets:数据集 dataframe格式
        ex_lis：不参与建模的特征，如id，时间切片等。 list格式
        min_samples：分箱时最小箱的样本占总比 numeric格式
        max_depth：决策树最大深度 numeric格式
        min_samples_leaf：决策树子节点最小样本个数 numeric格式
        min_samples_split：决策树划分前，父节点最小样本个数 numeric格式
        is_bin：是否进行卡方分箱 bool格式（True/False）
        '''
        self.datasets = datasets
        self.ex_lis = ex_lis
        self.dep = dep
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.is_bin = is_bin

        self.bins = 0

    def fit_plot(self):
        os.environ["PATH"] += os.pathsep + 'D:/Program Files/Graphviz2.38/bin'
        dtree = tree.DecisionTreeRegressor(max_depth=self.max_depth,
                                           min_samples_leaf=self.min_samples_leaf,
                                           min_samples_split=self.min_samples_split)

        x = self.datasets.drop(self.ex_lis, axis=1)
        y = self.datasets[self.dep]

        if self.is_bin:
            # 分箱
            combiner = toad.transform.Combiner()
            combiner.fit(x, y, method='chi', min_samples=self.min_samples)

            x_bin = combiner.transform(x)
            self.bins = combiner.export()
        else:
            x_bin = x.copy()

        dtree = dtree.fit(x_bin, y)

        df_bin = x_bin.copy()

        df_bin[self.dep] = y

        dot_data = StringIO()
        tree.export_graphviz(dtree, out_file=dot_data,
                             feature_names=x_bin.columns,
                             class_names=[self.dep],
                             filled=True, rounded=True,
                             special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

        return df_bin, self.bins, combiner, graph.create_png()



class DTreeCxPlot(object):

    def __init__(self, min_samples_leaf=200, min_samples_split=20, max_depth=4, criterion="gini", is_bin=True):
        self.tree_params = {"min_samples_leaf":min_samples_leaf, "min_samples_split":min_samples_split,
                            "max_depth":max_depth, "criterion":criterion}
        self.is_bin = is_bin
        self.bins = None

    def fit(self, df, label_identity="label", ignore_cols=None):

        ignore_cols = [label_identity] if not ignore_cols else [label_identity] + ignore_cols
        used_cols = [_ for _ in df.columns if _ not in ignore_cols]
        X_ = df.loc[:, used_cols]
        y = df.loc[:, label_identity]

        clf = tree.DecisionTreeClassifier(** self.tree_params)

        if self.is_bin:
            combiner = toad.transform.Combiner()
            combiner.fit(X_, y, method='chi', min_samples=0.1)
            X = combiner.transform(X_)
            self.bins = combiner.export()
        else:
            X = X_.copy()

        model = clf.fit(X, y)

        # dot_data = StringIO()
        dot_data = tree.export_graphviz(model, out_file=None,
                             feature_names=used_cols,
                             class_names=label_identity,
                             filled=True, rounded=True,
                             special_characters=True)
        self.graph = graphviz.Source(dot_data)


    def treeCxGraph(self, path=None):
        print("< -------------- >")
        self.graph.render("dtree", directory=path)


if __name__ == "__main__":

    df = pd.read_csv("../../data/credit/train.csv")

    obj = DTreeCxPlot(is_bin=False)

    obj.fit(df, label_identity="OVDU")

    obj.treeCxGraph()

