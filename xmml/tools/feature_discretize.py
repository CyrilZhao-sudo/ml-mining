# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/11/2

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from math import *
from scipy.stats import chisquare, chi2_contingency, chi2, fisher_exact, norm
from sklearn.tree import DecisionTreeClassifier


class Discretize:
    def __init__(self, df, cons_features, cate_features, label_identify="label", na_val=None, is_na_bin=False):
        self.raw = df
        self.cons_features = cons_features
        self.cate_features = cate_features
        self.label_identify = label_identify
        self.na_val = na_val  # 缺失值
        self.is_na_bin = is_na_bin  # 是否对缺失值单独设为一个箱

    def bin_freq_fit(self, qnt_num=10, min_block_size=16, contain_bound=False, worker=1, verbose=0):
        cons_bins = {}
        for cons_f in self.cons_features:
            bins = self.__bin_freq_single(self.raw, cons_f, qnt_num, min_block_size, self.na_val, contain_bound)
            # bins = Parallel(n_jobs=worker, verbose=verbose, )(delayed(self.__bin_freq_single)(self.raw, feat, qnt_num, min_block_size, self.spec_values) for feat in self.cons_features)
            cons_bins[cons_f] = bins
        return cons_bins

    def __bin_freq_single(self, df, feat_name, qnt_num, min_block_size, na_val=-1, contain_bound=False):
        _qnt_num = int(np.minimum(df[feat_name].nunique() / min_block_size, qnt_num))
        q = list(np.arange(0., 1., 1 / _qnt_num))
        bins = self.bin_freq_cuts(df[feat_name], q, na_val=na_val, contain_bound=contain_bound)
        return bins

    def bin_tree_fit(self, max_depth, min_samples_leaf, criterion="gini", **kwargs):
        cons_bins = {}
        for cons_feat in self.cons_features:
            df_x = self.raw[cons_feat]
            df_y = self.raw[self.label_identify]
            bins = self.__bin_tree_single(df_x, df_y, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                          criterion=criterion, **kwargs)
            cons_bins[cons_feat] = bins
        return cons_bins

    def __bin_tree_single(self, df_x, df_y, max_depth, min_samples_leaf=0.2, criterion="gini",
                          **kwargs):
        dt = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion,
                                    min_samples_leaf=min_samples_leaf, **kwargs)
        dt.fit(df_x.values.reshape(-1, 1), df_y)
        # print(dt.tree_.threshold)
        bins = dt.tree_.threshold[dt.tree_.threshold > 0]
        bins = np.sort(bins)
        return bins

    def bin_chi_fit(self, df, feat_name, is_continuous, threshold, p_val, init_freq_bins, sep="|"):
        '''
        默认对数值特征先进行等频，在进行卡方分箱合并；对类别直接进行卡方分箱合并
        :param df:
        :param feat_name:特征名称
        :param is_continuous: 是否是连续
        :param threshold: 每个bin的最小占比
        :param p_val: 当前bin和下一个bin的卡方检验p值
        :param init_freq_bins: 对连续特征初始化的等频分箱的个数
        :param sep: 类别分箱值合并的分隔符
        :return:
        '''
        if is_continuous:
            init_bins = self.__bin_freq_single(df, feat_name, qnt_num=init_freq_bins, min_block_size=10, na_val=self.na_val, contain_bound=False)
            cut_point = np.append(np.insert(init_bins, 0, -np.inf), np.inf)
            df["d_" + feat_name] = pd.cut(df[feat_name], bins=cut_point)
            raw_bins = self.chi_merge_bin(df, "d_" + feat_name, label_name=self.label_identify, threshold=threshold, p_val=p_val, sep=sep, is_factor=True)
            bins = np.unique(list(map(lambda x: x.left, raw_bins)) + list(map(lambda x: x.right, raw_bins)))
        else:
            if np.issubdtype(df[feat_name], np.number):
                df["d_" + feat_name] = df[feat_name].astype(str)
            else:
                df["d_" + feat_name] = df[feat_name]
            bins = self.chi_merge_bin(df, "d_" + feat_name, label_name=self.label_identify, threshold=threshold, p_val=p_val, sep=sep, is_factor=False)

        return bins

    def df_summary(self, df, d_feat_name, label_name, is_factor=False):
        # feat_name:cut之后或者是类别型变量
        df = df[[d_feat_name, label_name]]
        df_summary = df.groupby(d_feat_name)[label_name].agg([np.mean, np.count_nonzero, np.size]).rename(
            columns={"count_nonzero": "bad_obs", "size": "n", "mean": "bad_rate"}).reset_index()
        if is_factor:
            df_summary["arrange_col"] = df_summary[d_feat_name]
        else:
            df_summary["arrange_col"] = df_summary["bad_rate"].rank()
        df_summary = df_summary.sort_values(by=["arrange_col"]).reset_index(drop=True)
        df_summary["freq"] = df_summary["n"] / df_summary["n"].sum()
        df_summary["bad_rate_diff"] = df_summary["bad_rate"] - df_summary["bad_rate"].shift(1).fillna(0)
        df_summary["freq_sum"] = df_summary["freq"] + df_summary["freq"].shift(1).fillna(0)
        p_vals, l = [], len(df_summary)
        for i in range(l):
            if i == 0:
                res = self.__chi_square_pval(df_summary.loc[i].to_dict(), df_summary.loc[l - 1].to_dict())
            else:
                res = self.__chi_square_pval(df_summary.loc[i].to_dict(), df_summary.loc[i - 1].to_dict())
            p_vals.append(res)
        df_summary["p_val"] = p_vals
        return df_summary

    def __chi_square_pval(self, p, q):
        a = p.get("n") * p.get("bad_rate")
        b = p.get("n") - a
        c = q.get("n") * q.get("bad_rate")
        d = q.get("n") - c
        m = np.array([[a, b], [c, d]])
        if np.any(m < 100):
            _, p = fisher_exact(m)
        else:
            I, E1, dsq1 = p.get("n"), p.get("bad_rate"), p.get("V")
            J, E2, dsq2 = q.get("n"), q.get("bad_rate"), q.get("V")
            K = I + J
            Et = (E1 * I + E2 * J) / K
            dsq1 = 0 if dsq1 is None else dsq1
            dsq2 = 0 if dsq2 is None else dsq2
            dsqt = (dsq1 * (I - 1) + I * E1 * E1 + dsq2 * (J - 1) + J * E2 * E2 - K * Et * Et) / (K - 1)
            p = 2 * norm().pdf(-sqrt(I * J / K / dsqt) * abs(E1 - E2))

        return p

    def merge_bin_freq(self, df_xy, df_summary, d_feat_name, min_freq, threshold, label_name="label", sep="|",
                       is_factor=False):
        '''合并类别占比较低的箱'''
        while (min_freq < threshold and len(df_summary) > 2):
            min_freq_index = df_summary[df_summary["freq"] == min_freq].index[0]
            # min_freq_index = np.argmin(df_summary["freq"])
            if (min_freq_index == 0):
                ori_val = df_summary.head(2)[d_feat_name].to_list()
            elif (min_freq_index == len(df_summary) - 1):
                ori_val = df_summary.tail(2)[d_feat_name].to_list()
            elif (df_summary.loc[min_freq_index]["freq_sum"] > df_summary.loc[min_freq_index + 1]["freq_sum"]):
                ori_val = df_summary.loc[min_freq_index:min_freq_index + 1][d_feat_name].to_list()
            else:
                ori_val = df_summary.loc[min_freq_index - 1:min_freq_index][d_feat_name].to_list()
            df_xy[d_feat_name] = self.rename_bin_code(df_xy, ori_val, d_feat_name, sep, is_factor=is_factor)
            df_summary = self.df_summary(df_xy, d_feat_name, label_name, is_factor=is_factor)
            min_freq = min(df_summary["freq"])
            if (df_xy[d_feat_name].nunique() == 2):
                break
        return df_xy, df_summary

    def rename_bin_code(self, df, ori_value, d_feat_name, sep, is_factor):
        if is_factor:
            new_value = pd.Interval(min(map(lambda x: x.left, ori_value)), max(map(lambda x: x.right, ori_value)))
            replaced_series = self.__replace_category_values(df[d_feat_name], ori_values=ori_value,
                                                             new_values=new_value)
        else:
            if not isinstance(df[d_feat_name].dtype, object):
                df[d_feat_name] = df[d_feat_name].astype(str)
                ori_value = list(map(str, ori_value))
            new_value = "{}".format(sep).join(ori_value)
            replace_map = {k: new_value for k in ori_value}
            replaced_series = df[d_feat_name].map(replace_map)
        # TODO 对interval对象进行替换，变成nan
        return replaced_series

    def chi_merge_bin(self, df, d_feat_name, label_name, threshold=0.05, sep="|", p_val=0.05, is_factor=False):
        '''考虑数值特征对因子化和类别特征的非因子化，二者区别是一个按照值排序，另一个按照ovd排序'''
        df_xy = df[[d_feat_name, label_name]]
        n_unique = df[d_feat_name].nunique()
        if n_unique > 2:
            df_summary = self.df_summary(df_xy, d_feat_name, label_name, is_factor=is_factor)
            print(df_summary[[d_feat_name, "freq", "p_val"]])
            min_freq = min(df_summary.loc[:, "freq"])
            if min_freq < threshold:
                df_xy, df_summary = self.merge_bin_freq(df_xy, df_summary, d_feat_name, min_freq, threshold, label_name,
                                                        sep=sep, is_factor=is_factor)
            # min_bad_rate_diff = min(df_summary.loc[1:, "bad_rate_diff"])
            print(df_summary[[d_feat_name,"freq", "p_val"]])

            while (len(df_summary) > 2 and max(df_summary.loc[1:, "p_val"]) > p_val):  # todo 有点问题
                merge_bins_idx = np.argmax(df_summary.loc[1:, "p_val"]) + 1
                ori_values = df_summary.loc[merge_bins_idx - 1:merge_bins_idx, d_feat_name].to_list()
                df_xy[d_feat_name] = self.rename_bin_code(df_xy, ori_values, d_feat_name, sep=sep, is_factor=is_factor)

                df_summary = self.df_summary(df_xy, d_feat_name, label_name, is_factor=is_factor)
                print(df_summary[[d_feat_name,"freq", "p_val"]])
        else:
            df_summary = self.df_summary(df_xy, d_feat_name, label_name, is_factor=is_factor)
        bins = np.unique(df_summary.loc[:, d_feat_name])
        return bins

    @staticmethod
    def bin_freq_cuts(df_x, q, na_val=None, contain_bound=False):
        '''等频，分割点'''
        if not isinstance(df_x, pd.Series):
            try:
                df_x = pd.Series(df_x)
            except:
                raise TypeError("df_x must array like type!")
        if len(q) > 0 and q[-1] != 1. and contain_bound:
            q.append(1.)
        else:
            q.pop(0)
        if df_x.isna().sum():
            df_x = df_x[~pd.isna(df_x)]
            cuts = np.quantile(df_x, q)
            res = np.append(np.nan, np.unique(cuts))
        elif na_val is None:
            df_x = df_x[~pd.isna(df_x)]
            cuts = np.quantile(df_x, q)
            res = np.unique(cuts)
        else:
            df_x = df_x[df_x != na_val]
            cuts = np.quantile(df_x, q)
            res = np.append(na_val, np.unique(cuts))
        return res

    def __replace_category_values(self, x, ori_values, new_values):
        # 处理category的数据
        x_new = x.cat.remove_categories(ori_values).cat.add_categories(new_values).fillna(new_values)
        new_categories = sorted(x_new.cat.categories.to_list())
        x_new_reorder = x_new.cat.reorder_categories(new_categories, ordered=True)
        return x_new_reorder


def get_df_summary(df, feat_name, label_name, is_factor=False):
    df = df[[feat_name, label_name]]
    df_summary = df.groupby(feat_name)[label_name].agg([np.mean, np.count_nonzero, np.size]).rename(
        columns={"count_nonzero": "bad_obs", "size": "n", "mean": "bad_rate"}).reset_index()
    if is_factor:
        df_summary["arrange_col"] = df_summary[feat_name]
    else:
        df_summary["arrange_col"] = df_summary["bad_rate"].rank()
    df_summary = df_summary.sort_values(by=["arrange_col"]).reset_index(drop=True)
    df_summary["freq"] = df_summary["n"] / df_summary["n"].sum()
    df_summary["bad_rate_diff"] = df_summary["bad_rate"] - df_summary["bad_rate"].shift(1).fillna(0)
    df_summary["freq_sum"] = df_summary["freq"] + df_summary["freq"].shift(1).fillna(0)
    p_vals, l = [], len(df_summary)
    for i in range(l):
        if i == 0:
            res = chi_square_pval(df_summary.loc[i].to_dict(), df_summary.loc[l - 1].to_dict())
        else:
            res = chi_square_pval(df_summary.loc[i].to_dict(), df_summary.loc[i - 1].to_dict())
        p_vals.append(res)
    df_summary["p_val"] = p_vals
    return df_summary


def chi_square_pval(p, q):
    a = p.get("n") * p.get("bad_rate")
    b = p.get("n") - a
    c = q.get("n") * q.get("bad_rate")
    d = q.get("n") - c
    m = np.array([[a, b], [c, d]])
    if np.any(m < 100):
        _, p = fisher_exact(m)
    else:
        I, E1, dsq1 = p.get("n"), p.get("bad_rate"), p.get("V")
        J, E2, dsq2 = q.get("n"), q.get("bad_rate"), q.get("V")
        K = I + J
        Et = (E1 * I + E2 * J) / K
        dsq1 = 0 if dsq1 is None else dsq1
        dsq2 = 0 if dsq2 is None else dsq2
        dsqt = (dsq1 * (I - 1) + I * E1 * E1 + dsq2 * (J - 1) + J * E2 * E2 - K * Et * Et) / (K - 1)

        p = 2 * norm().cdf(-sqrt(I * J / K / dsqt) * abs(E1 - E2))

    return p


def __freq_bin_cuts(df_x, q, na_val=-1, contain_last=False):
    if not isinstance(df_x, pd.Series):
        try:
            df_x = pd.Series(df_x)
        except:
            raise TypeError("df_x must array like type!")
    if len(q) > 0 and q[-1] != 1. and contain_last:
        q = list(q) + [1.]  # 加不加上最后一个位置， 要看是否将大于最后一个位置是数看成一个箱; 最后一个位置的最大值
    if df_x.isna().sum():
        df_x = df_x[~pd.isna(df_x)]
        cuts = np.quantile(df_x, q)
        res = np.append(np.nan, np.unique(cuts))
    else:
        df_x = df_x[df_x != na_val]
        cuts = np.quantile(df_x, q)
        res = np.append(na_val, np.unique(cuts))
    return res


def __bin_tree_single(df_x, df_y, max_depth, min_samples_leaf=0.2, criterion="gini", min_samples_split=1, **kwargs):
    dt = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion, min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf, **kwargs)
    dt.fit(df_x.values.reshape(-1, 1), df_y)
    print(dt.tree_.threshold)
    bins = dt.tree_.threshold[dt.tree_.threshold > 0]
    bins = np.sort(bins)
    return bins
