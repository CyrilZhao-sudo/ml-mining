# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/11/2

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from math import *
from scipy.stats import chisquare,  chi2_contingency, chi2, fisher_exact, norm

class Discretize:
    def __init__(self, df, cons_features, cate_features, label_identify="label", spec_values=None):
        self.raw = df
        self.cons_features = cons_features
        self.cate_features = cate_features
        self.label_identify = label_identify
        self.spec_values = spec_values

    def bin_freq_fit(self, qnt_num=10, min_block_size=16):
        pass

    def bin_tree_fit(self):
        pass

    def bin_chi_fit(self):
        pass

    def __tree_discrete(self, params):
        pass

    def __df_summary(self, df, feat_name, label_name, is_factor=False):
        df = df[[feat_name, label_name]]
        if is_factor:
            pass
        df_summary = df.groupby(feat_name)[label_name].agg([np.mean, np.count_nonzero, np.size]).rename(
            columns={"count_nonzero": "bad_obs", "size": "n", "mean": "bad_rate"})
        df_summary["arrange_col"] = df_summary["bad_rate"].rank()
        df_summary = df_summary.sort_values(by=["arrange_col"]).reset_index()
        df_summary["freq"] = df_summary["n"] / df_summary["n"].sum()
        df_summary["bad_rate_diff"] = df_summary["bad_rate"] - df_summary["bad_rate"].shift(1).fillna(0)
        df_summary["freq_sum"] = df_summary["freq"] + df_summary["freq"].shift(1).fillna(0)
        p_vals, l = [], len(df_summary)
        for i in range(l):
            if i == 0:
                res = self.__chi_square_pval(df_summary.loc[i].to_dict(), df_summary.loc[l-1].to_dict())
            else:
                res = self.__chi_square_pval(df_summary.loc[i].to_dict(), df_summary.loc[i-1].to_dict())
            p_vals.append(res)
        df_summary["pval"] = p_vals

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




def get_df_summary(df, feat_name, label_name, is_factor=False):
    df = df[[feat_name, label_name]]
    if is_factor:
        pass
    df_summary = df.groupby(feat_name)[label_name].agg([np.mean, np.count_nonzero, np.size]).rename(columns={"count_nonzero":"bad_obs", "size":"n", "mean":"bad_rate"})
    df_summary["arrange_col"] = df_summary["bad_rate"].rank()
    df_summary = df_summary.sort_values(by=["arrange_col"]).reset_index()
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


df_summary = get_df_summary(data, "basicinfo_sex", "label")



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
        Et = (E1*I + E2*J) / K
        dsq1 = 0 if dsq1 is None else dsq1
        dsq2 = 0 if dsq2 is None else dsq2
        dsqt = (dsq1*(I-1) + I*E1*E1 + dsq2*(J-1) + J*E2*E2 - K*Et*Et) / (K-1)

        p = 2 * norm().cdf(-sqrt(I*J/K/dsqt) * abs(E1-E2))

    return p

chi_square_pval(df_summary.loc[0].to_dict(), df_summary.loc[1].to_dict())

