# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/11/2

import pandas as pd
import numpy as np
from joblib import Parallel, delayed


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

def get_df_summary(df, label_name, feat_name, is_factor=False):
    pass


from scipy.stats import chisquare,  chi2_contingency, chi2, fisher_exact
def chi_square_pval(b, o):
    a = b.get("N") * b.get("E")
    b = b.get("N") - a
    c = o.get("N") * o.get("E")
    d = o.get("N") - c
    pass

