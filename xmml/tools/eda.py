# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/11/2

import pandas as pd
import numpy as np
from joblib import Parallel, delayed

import time


def time_count(func):
    def inner(df, na_val=-1, threshold=0.2, ignore_cols=["id", "date", "label"], workers=1, verbose=0, display_result=True):
        s = time.time()
        res = func(df, na_val=na_val, threshold=threshold, ignore_cols=ignore_cols, workers=workers, verbose=verbose, display_result=display_result)
        e = time.time()
        print("delta time %d" % (e - s))
        return res

    return inner


def __coverage_rate(x, na_val=None):
    if isinstance(x, list) or isinstance(x, pd.Series):
        x = np.array(x)
    else:
        raise TypeError("x is must list like type!")
    if na_val is None:
        res = sum(np.isnan(x)) / len(x)
    else:
        res = sum(np.where(x == na_val, 1, 0)) / len(x)
    return 1. - res


@time_count
def calc_coverage_rate(df, na_val=-1, ignore_cols=["id", "date", "label"], workers=1, verbose=0, display_result=False,
                       threshold=0.2):
    df = df.copy()
    if ignore_cols:
        df = df.drop(columns=ignore_cols)
    results = Parallel(n_jobs=workers, verbose=verbose, )(delayed(__coverage_rate)(df[f], na_val) for f in df.columns)
    coverage_rate = pd.DataFrame(zip(df.columns, results), columns=["feature", "cov_rate"])
    if display_result:
        low_rate_cols = coverage_rate.loc[coverage_rate["cov_rate"] < 0.2, "feature"].tolist()
        print("the coverage rate is less than threshold {}, features num is {}".format(threshold, len(low_rate_cols)))
        print("the low coverage rate features are: {}".format(",".join(low_rate_cols)))
    return coverage_rate


# calc_coverage_rate(data, ignore_cols=["xiaomi_id", "pay_first_date", "label"], workers=4, verbose=1, display_result=True)


