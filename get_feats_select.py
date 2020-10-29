# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/10/28
import pandas as pd
import numpy as np

def get_lift_value(prob, label, p=0.1):
    df = pd.DataFrame({"prob":prob, "label":label})
    df["rank"] = df["prob"].rank()
    df.sort_values(by=['rank'], axis=0, ascending=False, inplace=True)
    n = int(p * len(df))
    base = np.mean(df["label"])
    p_observe = np.mean(df.iloc[:n]["label"])

    return p_observe / base

def print_rank_lift(prob, label, percentile=0.2):
    for i in np.arange(0.01, percentile+0.01, 0.01):
        tmp = get_lift_value(prob, label, i)
        print("rank {}, lift: {}".format(round(i, 2), tmp))



data = pd.read_csv("~/data/mi_data/data.csv")

def get_train_test_valid(df, test_size=0.2, valid_date_bound=None, date_name="pay_first_date", seed=123):
    from sklearn.model_selection import train_test_split
    if valid_date_bound is None:
        train, test = train_test_split(df, random_state=seed, test_size=test_size)
        valid = None
    else:
        tmp = df.loc[df[date_name] < valid_date_bound]
        train, test = train_test_split(tmp, random_state=seed, test_size=test_size)
        valid = df.loc[df[date_name] >= valid_date_bound]
    result = {"train":train, "test":test, "valid":valid}
    return result


from src.feature.NullImportance import NullImportance

null_importance = NullImportance(data, ignoreFeats=["xiaomi_id", "credit_time", "label", "basicinfo_provider", "basicinfo_group_provider", "repay_day", "pay_first_date", "cal_ovd_date",
                 "previous_day", "join_date", "date", "effective_time", "effective_day", "date_diff", "new_bairong_tag", "pay_cash_first_date", "pay_first_month",
                 "basicinfo_total_amount", "user_rate", "pay_cash_balance_p", "prin", "pay_cash_first_amount", "sum_over_balance"])

null_importance.fit()

select_features = null_importance.getImpFeatures(args=("mean", 75)).tolist()

null_importance.getInfoPlots()

train_test_valid = get_train_test_valid(data, valid_date_bound=20200501)

from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=100, learning_rate=0.01, max_depth=5, verbosity=1)
xgb.fit(train_test_valid["train"].loc[:,select_features], train_test_valid["train"]["label"], verbose=True,
        eval_set=[(train_test_valid["test"].loc[:,select_features], train_test_valid["test"]["label"])],
        eval_metric="logloss")

train_test_valid["valid"]["pred"] = xgb.predict_proba(train_test_valid["valid"].loc[:, select_features])[:, 1]

from sklearn.metrics import roc_auc_score

roc_auc_score(train_test_valid["valid"]["label"], train_test_valid["valid"]["pred"])

get_lift_value(train_test_valid["valid"]["pred"], train_test_valid["valid"]["label"], p=0.1)

print_rank_lift(train_test_valid["valid"]["pred"], train_test_valid["valid"]["label"], percentile=0.1)



