# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/11/2

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from math import *

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


from scipy.stats import chisquare,  chi2_contingency, chi2, fisher_exact, norm

def chi_square_pval(p, q):
    a = p.get("N") * p.get("E")
    b = p.get("N") - a
    c = q.get("N") * q.get("E")
    d = q.get("N") - c
    m = np.array([[a, b], [c, d]])
    if np.any(m < 100):
        _, p = fisher_exact(m)
    else:
        I, E1, dsq1 = p.get("N"), p.get("E"), p.get("V")
        J, E2, dsq2 = q.get("N"), q.get("E"), q.get("V")
        K = I + J
        Et = (E1*I + E2*J) / K
        if dsq1 is None:
            dsq1 = 0
        if dsq2 is None:
            dsq2 = 0
        dsqt = (dsq1*(I-1) + I*E1*E1 + dsq2*(J-1) + J*E2*E2 - K*Et*Et) / (K-1)
        p = 2 * norm().pdf(-sqrt(I*J/K/dsqt) * abs(E1-E2))

    return p



