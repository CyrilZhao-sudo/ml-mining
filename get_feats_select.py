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
    n_observe = np.sum(df.iloc[:n]["label"])
    return p_observe / base, n_observe

def print_rank_lift(prob, label, percentile=0.2):
    for i in np.arange(0.01, percentile+0.01, 0.01):
        tmp = get_lift_value(prob, label, i)
        print("rank {}, lift: {}, bad: {}".format(round(i, 2), tmp[0], tmp[1]))



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
from src.toolkit.loss_funcs import max_bad
from xgboost import XGBClassifier
xgb = XGBClassifier(n_estimators=200, learning_rate=0.005, max_depth=7,subsample=0.8, colsample_bytree=0.8,min_child_weight=3,
                    reg_alpha=1, reg_lambda=1, scale_pos_weight=10, verbosity=1, random_state=123)
xgb.fit(train_test_valid["train"].loc[:,select_features], train_test_valid["train"]["label"], verbose=True,
        eval_set=[(train_test_valid["train"].loc[:,select_features], train_test_valid["train"]["label"]),(train_test_valid["test"].loc[:,select_features], train_test_valid["test"]["label"])],
        eval_metric="logloss", early_stopping_rounds=30)

feature_importance = pd.DataFrame(data={"feature":select_features, "importance":xgb.feature_importances_}).sort_values(by=["importance"],ascending=False, ignore_index=True)
print(feature_importance)

train_test_valid["valid"]["pred"] = xgb.predict_proba(train_test_valid["valid"].loc[:, select_features])[:, 1]

from sklearn.metrics import roc_auc_score

roc_auc_score(train_test_valid["valid"]["label"], train_test_valid["valid"]["pred"])

get_lift_value(train_test_valid["valid"]["pred"], train_test_valid["valid"]["label"], p=0.05)

print_rank_lift(train_test_valid["valid"]["pred"], train_test_valid["valid"]["label"], percentile=0.1)

valid_202005 = train_test_valid["valid"][(train_test_valid["valid"]["pay_first_date"]>=20200501) & (train_test_valid["valid"]["pay_first_date"] < 20200601)]
roc_auc_score(valid_202005["label"], valid_202005["pred"])
valid_202006 = train_test_valid["valid"][(train_test_valid["valid"]["pay_first_date"]>=20200601) & (train_test_valid["valid"]["pay_first_date"] < 20200701)]
roc_auc_score(valid_202006["label"], valid_202006["pred"])
valid_202007 = train_test_valid["valid"][(train_test_valid["valid"]["pay_first_date"]>=20200701) & (train_test_valid["valid"]["pay_first_date"] < 20200801)]
roc_auc_score(valid_202007["label"], valid_202007["pred"])

valid_20200615 = train_test_valid["valid"][(train_test_valid["valid"]["pay_first_date"]>=20200501) & (train_test_valid["valid"]["pay_first_date"] < 20200615)]
roc_auc_score(valid_20200615["label"], valid_20200615["pred"])

print_rank_lift(valid_202005["pred"], valid_202005["label"], percentile=0.1)
print_rank_lift(valid_202006["pred"], valid_202006["label"], percentile=0.1)
print_rank_lift(valid_202007["pred"], valid_202007["label"], percentile=0.1)
print_rank_lift(valid_20200615["pred"], valid_20200615["label"], percentile=0.1)


import lightgbm as lgb

params = {
        'learning_rate': 0.015,
        'boosting_type': 'gbdt',
        'objective': 'huber',
        'metric': 'binary_logloss',
        'max_depth':7,
        'num_leaves': 64,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 3,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 30,
        'nthread': -1,
        'verbose': -1,
        'lambda_l1':1,
        'lambda_l2':1,
        'scale_pos_weight':5
    }

data_train = lgb.Dataset(data=train_test_valid["train"][select_features], label=train_test_valid["train"]["label"])
data_test = lgb.Dataset(data=train_test_valid["test"][select_features], label=train_test_valid["test"]["label"])
data_valid = lgb.Dataset(data=train_test_valid["valid"][select_features], label=train_test_valid["valid"]["label"])


clf = lgb.train(params, train_set=data_train, num_boost_round=500, valid_sets=[data_train, data_test], early_stopping_rounds=100, verbose_eval=10)


train_test_valid["valid"]["pred"] = clf.predict(train_test_valid["valid"].loc[:, select_features])







