# -*- coding: utf-8 -*-
# @File: NullImportance.py
# @Time: 7月 03, 2020



import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns



class NullImportance(object):

    def __init__(self, data, ignoreFeats=None, labelIdentity="label", categoryFeats=None):

        self.df = data.copy()
        self.ignoreFeats = ignoreFeats if ignoreFeats else [labelIdentity]
        self.labelIdentity = labelIdentity
        self.categoryFeats = categoryFeats if categoryFeats else []
        self.constanceFeats = [col for col in data.columns if col not in self.ignoreFeats and col not in self.categoryFeats]

        self.scoresDf = None
        self.scoreType = None



    def __getImportance(self, lgbParams=None, shuffle=False, seed=2020):

        features = self.constanceFeats + self.categoryFeats

        y = self.df[self.labelIdentity].copy().sample(frac=1.0, random_state=seed) if shuffle else self.df[self.labelIdentity].copy()

        dtrain = lgb.Dataset(self.df[features], y, free_raw_data=False, silent=True)

        if not lgbParams:
            lgbParams = {
                'objective': 'binary',
                'boosting_type': 'rf',
                'subsample': 0.623,
                'colsample_bytree': 0.7,
                'num_leaves': 127,
                'max_depth': 8,
                'seed': seed,
                'bagging_freq': 1,
                'n_jobs': 4
            }
        # num_boost_round = 100 * (num_feats / sqrt(num_feats) * depth))
        clf = lgb.train(lgbParams, train_set=dtrain, num_boost_round=100, categorical_feature=self.categoryFeats if self.categoryFeats else "auto")

        importanceDf = pd.DataFrame()
        importanceDf["feature"] = features
        importanceDf["importance_gain"] = clf.feature_importance(importance_type="gain")
        importanceDf["importance_split"] = clf.feature_importance(importance_type="split")
        importanceDf["auc_score"] = roc_auc_score(y_true=y, y_score=clf.predict(self.df[features]))

        return importanceDf


    def fit(self, iterNums=30, lgbParams=None, seed=123):
        self.importanceDf = self.__getImportance(seed=seed)

        nullImportanceDf = pd.DataFrame()

        for i in range(iterNums):
            tmp = self.__getImportance(lgbParams, shuffle=True, seed=seed)
            tmp["run"] = i + 1
            nullImportanceDf = pd.concat([nullImportanceDf, tmp], axis=0)

        self.nullImportanceDf = nullImportanceDf

        return self


    def __getFeatureScores(self, percentile=75, is_mean=False):

        featureScores = []

        features = self.importanceDf["feature"].unique()

        for f in features:
            f_act_imp_gain = self.importanceDf.loc[self.importanceDf["feature"] == f, "importance_gain"].mean()
            f_null_imp_gain = self.nullImportanceDf.loc[self.nullImportanceDf["feature"] == f, "importance_gain"].values

            f_null_imp_split = self.nullImportanceDf.loc[self.nullImportanceDf["feature"] == f, "importance_split"].values
            f_act_imp_split = self.importanceDf.loc[self.importanceDf["feature"] == f, "importance_split"].mean()

            if is_mean:
                gain_score = np.log(f_act_imp_gain / (1 + np.mean(f_null_imp_gain)) + 1e-10)
                split_score = np.log(f_act_imp_split / (1 + np.mean(f_null_imp_split)) + 1e-10)
            else:
                gain_score = np.log(f_act_imp_gain / (1 + np.percentile(f_null_imp_gain, percentile)) + 1e-10)
                split_score = np.log(f_act_imp_split / (1 + np.percentile(f_null_imp_split, percentile)) + 1e-10)

            featureScores.append((f, gain_score, split_score))

        scoresDf = pd.DataFrame(featureScores, columns=["feature", "gain_score", "split_score"])
        self.scoresDf = scoresDf.copy()
        return scoresDf

    def getImpFeatures(self, filterType="both", threshold=0, args=()):
        # 不同的评价方式 nullImpType percentile=75/max/mean
        (nullType, percentile) = args
        if nullType == "percentile":
            scoresDf = self.__getFeatureScores(percentile)
        elif nullType == "max":
            scoresDf = self.__getFeatureScores(percentile)
        elif nullType == "mean":
            scoresDf = self.__getFeatureScores(is_mean=True)
        else:
            raise Exception("> {} null importance type is not valid!".format(nullType))
        #print(scoresDf)
        if filterType == "both":
            selected_feature = scoresDf.loc[(scoresDf["gain_score"] > threshold) & (scoresDf["split_score"] > threshold), "feature"].unique()
        elif filterType == "split":
            selected_feature = scoresDf.loc[scoresDf["split_score"] > threshold, "feature"].unique()
        else:
            selected_feature = scoresDf.loc[scoresDf["gain_score"] > threshold, "feature"].unique()

        return selected_feature


    def getInfoPlots(self):
        if self.scoresDf is None:
            raise Exception("please get feature score dataFrame first!")

        plt.figure(figsize=(14, 14))
        gs = gridspec.GridSpec(1, 2)
        # Plot Split importances
        ax = plt.subplot(gs[0, 0])
        sns.barplot(x='split_score', y='feature', data=self.scoresDf.sort_values('split_score', ascending=False),
                    ax=ax)
        ax.set_title('Feature scores wrt split importances', fontweight='bold', fontsize=14)
        # Plot Gain importances
        ax = plt.subplot(gs[0, 1])
        sns.barplot(x='gain_score', y='feature', data=self.scoresDf.sort_values('gain_score', ascending=False),
                    ax=ax)
        ax.set_title('Feature scores wrt gain importances', fontweight='bold', fontsize=14)
        plt.tight_layout()

        plt.show()


if __name__ == "__main__":

    data = pd.read_csv("./data/train.csv")
    newColunms = {"年龄":"age", "性别":"sex", "婚姻状况":"marry", "同盾":"td", "平台":"platform", "商品总价":"price",
                  "首付金额":"firstPayAmount", "贷款金额":"Amount", "分期期数":"periods", "是否保险":"x1", "是否随心包":"x2",
                  "住房类型":"houseType", "手机使用时长":"phoneUsedTime", "是否实名":"x3", "受教育程度":"degree", "是否办理过分期业务":"x4",
                  "月收入":"income", "芝麻信用分":"score"}
    data.rename(newColunms, inplace=True, axis=1)

    for cate in ["sex","marry","x1", "platform", "x3","degree","x4", "x2", "houseType"]:
        data[cate], _ = pd.factorize(data[cate])

    nullImpSelect = NullImportance(data, ignoreFeats=None, labelIdentity="OVDU",
                                   categoryFeats=["sex","marry","x1", "platform", "x3","degree","x4", "x2", "houseType"])

    nullImpSelect.fit(iterNums=1)

    nullImpSelect.getImpFeatures(filterType="gain_score",args=("mean", 75))



    nullImpSelect.getInfoPlots()



