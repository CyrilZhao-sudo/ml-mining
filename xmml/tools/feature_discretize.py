# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/11/2

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from math import *
from scipy.stats import chisquare, chi2_contingency, chi2, fisher_exact, norm
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import itertools
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")


class Discretize:
    def __init__(self, train_df, cons_features, cate_features, label_identify="label", na_val=None, is_na_bin=False,
                 cache_stat=False, sep="|"):
        self.raw = train_df  # train dataFrame
        self.cons_features = cons_features if cons_features else []
        self.cate_features = cate_features if cate_features else []
        self.label_identify = label_identify
        self.na_val = na_val  # 缺失值
        self.is_na_bin = is_na_bin  # 是否对缺失值单独设为一个箱
        self.sep = sep
        self.cons_bins = None
        self.cate_bins = None
        self.le_encoders = {}
        self.onehot_encoders = {}
        self.cache_stat = cache_stat
        self.woe_stats = {}
        self._iv = []
        self.top_cate = 10
        self._cate_mode = {}

    def bin_freq_fit(self, qnt_num=10, min_block_size=16, contain_bound=False, worker=1, verbose=0,
                     category_method="chi", threshold=0.05, p_val=0.05):
        self.reset_status(reset_cate_bins=False)
        cons_bins, cate_bins = {}, {}
        for cons_f in tqdm(self.cons_features):
            bins = self._bin_freq_single(self.raw, cons_f, qnt_num, min_block_size, self.na_val, contain_bound, is_continuous=True, calc_stat=True)
            # bins = Parallel(n_jobs=worker, verbose=verbose, )(delayed(self._bin_freq_single)(self.raw, feat, qnt_num, min_block_size, self.spec_values) for feat in self.cons_features)
            cons_bins[cons_f] = bins
        self.cons_bins = cons_bins.copy()

        for cate_f in tqdm(self.cate_features):
            df_xy = self.raw[[cate_f, self.label_identify]]
            bins = self.bin_category_single(df_xy, cate_f, threshold=threshold, p_val=p_val, sep=self.sep,
                                            method=category_method)
            cate_bins[cate_f] = bins
        self.cate_bins = cate_bins.copy()

    def _bin_freq_single(self, df, feat_name, qnt_num, min_block_size, na_val=-1, contain_bound=False, is_continuous=True, calc_stat=False):
        nunique = df[feat_name].nunique()
        _qnt_num = int(np.minimum( nunique / min_block_size, qnt_num))
        _qnt_num = _qnt_num if _qnt_num > 0 else nunique
        q = list(np.arange(0., 1., 1 / _qnt_num))
        bins = self.bin_freq_cuts(df[feat_name], q, na_val=na_val, contain_bound=contain_bound)
        if calc_stat:
            stat = self.__calc_stat_single(df[[feat_name, self.label_identify]], bins, feat_name, is_continuous=is_continuous,
                                           interval_str=True)
            self._iv.append((feat_name, np.sum(stat["iv_i"])))
        return bins

    def bin_category_single(self, df_xy, feat_name, threshold, p_val, sep, method):
        n_unique = len(np.unique(df_xy[feat_name]))
        if n_unique > (len(df_xy) * 0.005):
            vc = df_xy[feat_name].value_count(ascending=False)[:self.top_cate]
            raise NotImplementedError
        else:
            if method == "chi":
                bins = self.bin_chi_single(df_xy, feat_name, is_continuous=False, threshold=threshold, p_val=p_val,
                                           init_freq_bins=10, sep=sep)
            elif method == "freq_merge":
                df_summary = self.df_summary(df_xy, feat_name, self.label_identify)
                min_freq = min(df_summary["freq"])
                bins = self.merge_bin_freq(df_xy, df_summary, feat_name,
                                           min_freq=min_freq, threshold=threshold, label_name=self.label_identify,
                                           sep=self.sep, return_bins=True)
            else:
                raise ValueError("")
        return bins

    def bin_tree_fit(self, max_depth, min_samples_leaf, criterion="gini", **kwargs):
        self.reset_status(reset_cate_bins=False)
        cons_bins = {}
        for cons_feat in tqdm(self.cons_features):
            print("dealing {}".format(cons_feat))
            df_xy = self.raw[[cons_feat, self.label_identify]]
            bins = self.__bin_tree_single(df_xy, cons_feat, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                          criterion=criterion, **kwargs)
            cons_bins[cons_feat] = bins
        self.cons_bins = cons_bins.copy()


    def __bin_tree_single(self, df_xy, feat_name, max_depth, min_samples_leaf=0.2, criterion="gini",
                          **kwargs):
        dt = DecisionTreeClassifier(max_depth=max_depth, criterion=criterion,
                                    min_samples_leaf=min_samples_leaf, **kwargs)
        x, y = df_xy[feat_name], df_xy[self.label_identify]
        dt.fit(x.values.reshape(-1, 1), y)
        bins = dt.tree_.threshold[dt.tree_.threshold > 0]
        bins = np.sort(bins)
        if len(bins) > 0:
            stat = self.__calc_stat_single(df_xy[[feat_name, self.label_identify]], bins, feat_name, is_continuous=True,
                                           interval_str=True)
            self._iv.append((feat_name, np.sum(stat["iv_i"])))
        else:
            print("feature: {}  is error")
            return None

        return bins

    def bin_chi_fit(self, threshold=0.05, p_val=0.05, init_freq_bins=10, category_method="chi"):
        self.reset_status(reset_cate_bins=False)
        cons_bins, cate_bins = {}, {}
        for cons_f in tqdm(self.cons_features):
            print("deal {}".format(cons_f))
            bins = self.bin_chi_single(self.raw, cons_f, is_continuous=True, threshold=threshold, p_val=p_val, init_freq_bins=init_freq_bins, sep=self.sep)
            # bins = Parallel(n_jobs=worker, verbose=verbose, )(delayed(self._bin_freq_single)(self.raw, feat, qnt_num, min_block_size, self.spec_values) for feat in self.cons_features)
            cons_bins[cons_f] = bins
        self.cons_bins = cons_bins.copy()

        for cate_f in tqdm(self.cate_features):
            df_xy = self.raw[[cate_f, self.label_identify]]
            bins = self.bin_category_single(df_xy, cate_f, threshold=threshold, p_val=p_val, sep=self.sep,
                                            method=category_method)
            cate_bins[cate_f] = bins
        self.cate_bins = cate_bins.copy()

    def bin_chi_single(self, df, feat_name, is_continuous, threshold, p_val, init_freq_bins, sep="|"):
        '''
        默认对数值特征先进行等频，在进行卡方分箱合并；对类别直接进行卡方分箱合并
        :param df: TODO df_xy or df ?
        :param feat_name:特征名称
        :param is_continuous: 是否是连续
        :param threshold: 每个bin的最小占比
        :param p_val: 当前bin和下一个bin的卡方检验p值
        :param init_freq_bins: 对连续特征初始化的等频分箱的个数
        :param sep: 类别分箱值合并的分隔符
        :return:
        '''
        if is_continuous:
            init_bins = self._bin_freq_single(df, feat_name, qnt_num=init_freq_bins, min_block_size=10,
                                               na_val=self.na_val, contain_bound=False, calc_stat=False)
            cut_point = np.append(np.insert(init_bins, 0, -np.inf), np.inf)
            df["d_" + feat_name] = pd.cut(df[feat_name], bins=cut_point)
            raw_bins = self.chi_merge_bin(df, "d_" + feat_name, label_name=self.label_identify, threshold=threshold,
                                          p_val=p_val, sep=sep, is_factor=True)
            bins = np.unique(list(map(lambda x: x.left, raw_bins)) + list(map(lambda x: x.right, raw_bins)))
        else:
            if np.issubdtype(df[feat_name], np.number):
                df["d_" + feat_name] = df[feat_name].astype(str)
            else:
                df["d_" + feat_name] = df[feat_name]
            bins = self.chi_merge_bin(df, "d_" + feat_name, label_name=self.label_identify, threshold=threshold,
                                      p_val=p_val, sep=sep, is_factor=False)

        stat = self.__calc_stat_single(df[[feat_name, self.label_identify]], bins, feat_name, is_continuous,
                                       interval_str=True)
        self._iv.append((feat_name, np.sum(stat["iv_i"])))

        return bins

    def __calc_stat_single(self, df_xy, bins, feat_name, is_continuous, interval_str):
        df_xy_dis = self.df_x_discretize(df_xy, bins, feat_name, is_continuous, interval_str)
        stat = self.calc_woe_single(df_xy_dis, "d_" + feat_name)
        if self.cache_stat:
            self.woe_stats[feat_name] = stat
        return stat

    def woe_transform(self, data=None):
        '''默认返回的是训练集的woe. 当传入data时,基于训练集对其进行woe转换'''
        df = self.raw.copy()
        woe_df = pd.DataFrame()
        if self.cons_bins:
            for cons_feat, cons_bins in self.cons_bins.items():
                trans_df = data[[cons_feat]] if isinstance(data, pd.DataFrame) else None
                woe_x = self.woe_transform_single(df[[cons_feat, self.label_identify]], bins=cons_bins,
                                                  feat_name=cons_feat, is_continuous=True, transform_df=trans_df)
                woe_df[cons_feat] = woe_x
        if self.cate_bins:
            for cate_feat, cate_bins in self.cate_bins.items():
                trans_df = data[[cate_feat]] if isinstance(data, pd.DataFrame) else None
                woe_x = self.woe_transform_single(df[[cate_feat, self.label_identify]], bins=cate_bins,
                                                  feat_name=cate_feat, is_continuous=False, transform_df=trans_df)
                woe_df[cate_feat] = woe_x
        return woe_df

    def woe_transform_single(self, df_xy, bins, feat_name, is_continuous=True, transform_df=None):
        if self.woe_stats.get(feat_name) is not None:
            stat = self.woe_stats.get(feat_name)
            df_xy_dis = self.df_x_discretize(df_xy, bins, feat_name, is_continuous, interval_str=True)
        else:
            bins = bins.copy()
            df_xy_dis = self.df_x_discretize(df_xy, bins, feat_name, is_continuous, interval_str=True)
            stat = self.calc_woe_single(df_xy_dis, "d_" + feat_name)
        woe_replace_map = {k: v for (_, k, v) in stat[["d_" + feat_name, "woe"]].itertuples()}
        if transform_df is None:
            woe_x = df_xy_dis["d_" + feat_name].map(woe_replace_map)
        else:
            trans_df_dis = self.df_x_discretize(transform_df, bins, feat_name, is_continuous, interval_str=True)
            woe_x = trans_df_dis["d_" + feat_name].map(woe_replace_map)
        return woe_x

    def df_x_discretize(self, df_x, bins, feat_name, is_continuous, interval_str=False):
        '''对单个特征进行分箱映射成每个箱，同时返回对应分箱之后映射的列'''
        if is_continuous:
            if not (bins[0] == -np.infty and bins[-1] == np.infty):
                bins = np.append(np.insert(bins, 0, -np.inf), np.inf)
            df_x["d_" + feat_name] = pd.cut(df_x[feat_name], bins=bins)
            if interval_str:
                df_x["d_" + feat_name] = df_x["d_" + feat_name].astype(str)
        else:
            if np.issubdtype(df_x[feat_name], np.number):
                df_x[feat_name] = df_x[feat_name].astype(str)
            cate_replace_map = self.replace_cate_values(df_x[feat_name], bins, feat_name)
            df_x["d_" + feat_name] = df_x[feat_name].map(cate_replace_map)
        return df_x

    def replace_cate_values(self, x, bins, feat_name, sep="|", na_val=-1):
        '''默认代替unknown的顺序:na -> unknown -> mode'''
        x_unique = np.unique(x)
        ori_unique = np.unique(list(itertools.chain.from_iterable(map(lambda x: x.split(sep), bins))))
        oof = set(x_unique).difference(set(ori_unique))
        mode = self._cate_mode.get(feat_name)
        oof_bins = {"mode": mode}
        replace_map = {}
        for b in bins:
            if "unknown" in b:
                oof_bins["unknown"] = b
            if str(na_val) in b:
                oof_bins[str(na_val)] = b
            for x in x_unique:
                if x in b:
                    replace_map.update({x: b})
        if len(oof_bins) == 0 and len(oof) != 0:
            raise ValueError
        elif len(oof_bins) > 0 and len(oof) != 0:
            for o in oof:
                replace_map.update({o: oof_bins.get(str(na_val), oof_bins.get("unknown", mode))})
        else:
            pass
        return replace_map

    def calc_woe_single(self, df_xy, d_feat_name):
        stat = df_xy.groupby([d_feat_name])[self.label_identify].agg([np.mean, np.count_nonzero, np.size]).rename(
            columns={"count_nonzero": "bad_obs", "size": "n", "mean": "bad_rate"}).reset_index()
        total = stat["n"].sum()
        stat["freq"] = stat["n"] / total
        stat["good_obs"] = stat["n"] - stat["bad_obs"]
        t_good = np.maximum(stat["good_obs"].sum(), 0.5)
        t_bad = np.maximum(stat["bad_obs"].sum(), 0.5)
        stat["woe"] = stat.apply(self._bucket_woe, axis=1) + np.log(t_bad / t_good)
        stat["iv_i"] = (stat["good_obs"] / t_good - stat["bad_obs"] / t_bad) * stat["woe"]
        return stat

    def onehot_transform(self, data=None):
        df = self.raw.copy() if not data else data.copy()
        onehot_df = []
        if self.cons_bins:
            for cons_feat, cons_bins in self.cons_bins.items():
                onehot_x_sparse = self.onehot_encoder_single(df[[cons_feat]], bins=cons_bins,
                                                             feat_name=cons_feat, is_continuous=True)
                onehot_x_df = pd.DataFrame(data=onehot_x_sparse.toarray(),
                                           columns=[cons_feat + "_x{}".format(i) for i in
                                                    range(onehot_x_sparse.shape[1])])
                onehot_df.append(onehot_x_df)
        if self.cate_bins:
            for cate_feat, cate_bins in self.cate_bins.items():
                onehot_x_sparse = self.onehot_encoder_single(df[[cate_feat]], bins=cate_bins,
                                                             feat_name=cate_feat, is_continuous=False)
                onehot_x_df = pd.DataFrame(data=onehot_x_sparse.toarray(),
                                           columns=[cate_feat + "_x{}".format(i) for i in
                                                    range(onehot_x_sparse.shape[1])])
                onehot_df.append(onehot_x_df)
        onehot_df = pd.concat(onehot_df, axis=1)
        return onehot_df

    def onehot_encoder_single(self, df_x, bins, feat_name, is_continuous):
        bins = bins.copy()
        # if is_continuous:
        #     if not (bins[0] == -np.infty and bins[-1] == np.infty):
        #         bins = np.append(np.insert(bins, 0, -np.inf), np.inf)
        #     df_x["d_" + feat_name] = pd.cut(df_x[feat_name], bins=bins)
        # else:
        #     if np.issubdtype(df_x[feat_name], np.number):
        #         df_x[feat_name] = df_x[feat_name].astype(str)
        #     cate_replace_map = self.replace_cate_values(df_x[feat_name], bins)
        #     df_x["d_" + feat_name] = df_x[feat_name].map(cate_replace_map)
        df_x_dis = self.df_x_discretize(df_x, bins, feat_name, is_continuous, interval_str=False)
        if feat_name in self.onehot_encoders:
            oe = self.onehot_encoders[feat_name]
        else:
            oe = OneHotEncoder(dtype=np.float32, drop='if_binary')
            oe.fit(df_x_dis[["d_" + feat_name]])
            self.onehot_encoders[feat_name] = oe
        x_oe_sparse = oe.transform(df_x_dis[["d_" + feat_name]])
        return x_oe_sparse

    def label_encode_transform(self, data=None):
        df = self.raw.copy() if not data else data.copy()
        le_df = pd.DataFrame()
        if self.cons_bins:
            for cons_feat, cons_bins in self.cons_bins.items():
                le_x = self.label_encoder_single(df[[cons_feat]], bins=cons_bins,
                                                 feat_name=cons_feat, is_continuous=True)
                le_df[cons_feat] = le_x
        if self.cate_bins:
            for cate_feat, cate_bins in self.cate_bins.items():
                le_x = self.label_encoder_single(df[[cate_feat]], bins=cate_bins,
                                                 feat_name=cate_feat, is_continuous=False)
                le_df[cate_feat] = le_x
        return le_df

    def label_encoder_single(self, df_x, bins, feat_name, is_continuous):
        bins = bins.copy()
        # if is_continuous:
        #     if not (bins[0] == -np.infty and bins[-1] == np.infty):
        #         bins = np.append(np.insert(bins, 0, -np.inf), np.inf)
        #     df_x["d_" + feat_name] = pd.cut(df_x[feat_name], bins=bins)
        # else:
        #     if np.issubdtype(df_x[feat_name], np.number):
        #         df_x[feat_name] = df_x[feat_name].astype(str)
        #     cate_replace_map = self.replace_cate_values(df_x[feat_name], bins)
        #     df_x["d_" + feat_name] = df_x[feat_name].map(cate_replace_map)
        df_x_dis = self.df_x_discretize(df_x, bins, feat_name, is_continuous, interval_str=False)
        if feat_name in self.le_encoders:
            le = self.le_encoders[feat_name]
        else:
            le = LabelEncoder()
            le.fit(df_x_dis["d_" + feat_name])
            self.le_encoders[feat_name] = le
        x_le = le.transform(df_x_dis["d_" + feat_name])
        return x_le

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
            print(m)
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
                       is_factor=False, return_bins=False):
        '''合并类别占比较低的箱'''
        while (min_freq < threshold and len(df_summary) > 2):
            # print(df_summary.drop(columns=["freq_sum", "bad_rate_diff"]))
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
        if return_bins:
            feat_name = d_feat_name[2:]
            argmax_index = np.argmax(df_summary["freq"])
            self._cate_mode[feat_name] = df_summary.loc[argmax_index, d_feat_name]
            bins = np.unique(df_summary.loc[:, d_feat_name])
            return bins
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
            replaced_series = df[d_feat_name].apply(lambda x: replace_map.get(x, x))

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
            print(df_summary[[d_feat_name, "freq", "p_val"]])

            while (len(df_summary) > 2 and max(df_summary.loc[1:, "p_val"]) > p_val):  # todo 有点问题
                merge_bins_idx = np.argmax(df_summary.loc[1:, "p_val"]) + 1
                ori_values = df_summary.loc[merge_bins_idx - 1:merge_bins_idx, d_feat_name].to_list()
                df_xy[d_feat_name] = self.rename_bin_code(df_xy, ori_values, d_feat_name, sep=sep, is_factor=is_factor)

                df_summary = self.df_summary(df_xy, d_feat_name, label_name, is_factor=is_factor)
                print(df_summary[[d_feat_name, "freq", "p_val"]])
        else:
            df_summary = self.df_summary(df_xy, d_feat_name, label_name, is_factor=is_factor)
        bins = np.unique(df_summary.loc[:, d_feat_name])
        if not is_factor:
            argmax_index = np.argmax(df_summary["freq"])
            feat_name = d_feat_name[2:]
            self._cate_mode[feat_name] = df_summary.loc[argmax_index, d_feat_name]
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

    @staticmethod
    def _bucket_woe(x):
        bad_i = x['bad_obs']
        good_i = x['good_obs']
        bad_i = 0.5 if bad_i == 0 else bad_i
        good_i = 0.5 if good_i == 0 else good_i
        return np.log(good_i / bad_i)

    def __replace_category_values(self, x, ori_values, new_values):
        # 处理category/interval的数据
        x_new = x.cat.remove_categories(ori_values).cat.add_categories(new_values).fillna(new_values)
        new_categories = sorted(x_new.cat.categories.to_list())
        x_new_reorder = x_new.cat.reorder_categories(new_categories, ordered=True)
        return x_new_reorder

    @property
    def iv_df(self):
        if self._iv:
            return pd.DataFrame(self._iv, columns=["feature", "iv"])
        else:
            return None

    def reset_status(self, reset_cate_bins=True):
        print("reset status.")
        self._iv = []
        self._cate_mode = {}
        self.cons_bins = None
        if reset_cate_bins:
            self.cate_bins = None
