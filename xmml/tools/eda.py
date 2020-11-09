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


# @time_count
def calc_coverage_rate(df, na_val=-1, ignore_cols=["id", "date", "label"], workers=1, verbose=0, display_result=False,
                       threshold=0.2, method="all"):
    df = df.copy()
    if ignore_cols:
        df = df.drop(columns=ignore_cols)
    if method == "all":
        results = Parallel(n_jobs=workers, verbose=verbose, )(delayed(__coverage_rate)(df[f], na_val) for f in df.columns)
        coverage_rate = pd.DataFrame(zip(df.columns, results), columns=["feature", "cov_rate"])
    elif method == "month":
        months = df["month"].unique()
        coverage_rate_list = []
        for month in months:
            df_by_month = df[df["month"] == month].drop(columns=["month"])
            cov_rate_month = Parallel(n_jobs=workers, verbose=verbose, )(delayed(__coverage_rate)(df_by_month[f], na_val) for f in df_by_month.columns)
            cov_rate_month_df = pd.DataFrame(zip(df_by_month.columns, cov_rate_month), columns=["feature", "cov_rate"])
            cov_rate_month_df["month"] = month
            coverage_rate_list.append(cov_rate_month_df)
        coverage_rate = pd.concat(coverage_rate_list, axis=0, ignore_index=True)
    if display_result:
        low_rate_cols = coverage_rate.loc[coverage_rate["cov_rate"] < 0.2, "feature"].tolist()
        print("the coverage rate is less than threshold {}, features num is {}".format(threshold, len(low_rate_cols)))
        print("the low coverage rate features are: {}".format(",".join(low_rate_cols)))
    return coverage_rate


# calc_coverage_rate(data, ignore_cols=["xiaomi_id", "pay_first_date", "label"], workers=4, verbose=1, display_result=True)

ignore_cols = ["xiaomi_id", "credit_time", "label", "basicinfo_provider", "basicinfo_group_provider", "repay_day", "pay_first_date", "cal_ovd_date",
                 "previous_day", "join_date", "date", "effective_time", "effective_day", "date_diff", "new_bairong_tag", "pay_cash_first_date", "pay_first_month",
                 "basicinfo_total_amount", "user_rate", "pay_cash_balance_p", "prin", "pay_cash_first_amount", "sum_over_balance", "alf_apirisk_time_slope_d30_sum",
                 "mi_user_model_youpin_v2"]
data = pd.read_csv("/Users/hudoudou/zhaochen/data/data.csv")
data['month'] = (data['pay_first_date'] / 100).astype(int)

cov_rate = calc_coverage_rate(data, ignore_cols=ignore_cols, workers=3, method="month")

model_feats = ["loan_amount_ratio", "prm_level1_high_risk", "prm_level2_high_risk", "lrm_high_risk",
               "als_m12_id_caon_orgnum", "tvbox_exist", "last_30_days_total_count", "last_30_days_v1_user_profile_info_count",
               "last_30_days_v1_home_page_count_ratio", "pay_cash_first_duration", "mi_user_model_appusage_v2",
               "mi_user_model_browser_v2", "mi_user_model_account", "mi_user_model_mimall", "mi_user_model_sms_v2",
               "scoreconson", "scoreafconsoff", "scorepdl", "alf_freq_max_d15", "alf_freq_max_d15", "risk_score"]

for col in data.columns:
    if col in model_feats:
        t = cov_rate[cov_rate['feature'] == col].sort_values(by=["month"])
        print(t)
        time.sleep(1)