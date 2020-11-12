# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/10/28
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

#显示所有列
pd.set_option('display.max_columns', 20)
#显示所有行
pd.set_option('display.max_rows', 100)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

def get_lift_value(prob, label, p=0.1):
    df = pd.DataFrame({"prob":prob, "label":label})
    df.sort_values(by=['prob'], axis=0, ascending=False, inplace=True, ignore_index=True)
    reject_rank, base_default_rate = int(p * len(df)), np.mean(df["label"])
    prob_threshold= df.loc[reject_rank-1, "prob"]
    reject_df = df.loc[df["prob"]>=prob_threshold, :]
    reject_bad_rate = np.mean(reject_df["label"])
    reject_bad_n = np.sum(reject_df["label"])
    reject_n = len(reject_df)
    return reject_bad_rate / base_default_rate, reject_bad_n, reject_n

def print_rank_lift(prob, label, percentile=0.2):
    for i in np.arange(0.01, percentile+0.01, 0.01):
        tmp = get_lift_value(prob, label, i)
        print("{}, lift: {}, bad: {}, n: {}".format(round(i, 3), tmp[0], tmp[1], tmp[2]))


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



def run_simple_lgb(train_test_valid, select_features, cate_features=None, label_name="label"):
    train, test, valid = train_test_valid["train"], train_test_valid["test"], train_test_valid["valid"]

    data_train = lgb.Dataset(data=train[select_features], label=train[label_name])
    data_test = lgb.Dataset(data=test[select_features], label=test[label_name])
    # data_valid = lgb.Dataset(data=train_test_valid["valid"][select_features], label=train_test_valid["valid"][label_name])

    params = {
        'learning_rate': 0.02,
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'max_depth': 5,
        'num_leaves': 36,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 3,
        'seed': 1,
        'bagging_seed': 1,
        'feature_fraction_seed': 7,
        'min_data_in_leaf': 30,
        'nthread': -1,
        'verbose': -1,
        'lambda_l1': 5,
        'lambda_l2': 5,
        'scale_pos_weight': 1
    }
    clf = lgb.train(params, train_set=data_train, num_boost_round=500, valid_sets=[data_train, data_test],
                    early_stopping_rounds=100, verbose_eval=10,categorical_feature=cate_features if cate_features else "auto")

    valid["pred"] = clf.predict(valid.loc[:, select_features])
    train["pred"] = clf.predict(train.loc[:, select_features])
    test["pred"] = clf.predict(test.loc[:, select_features])
    train_auc, test_auc, valid_auc = roc_auc_score(train[label_name], train["pred"]), roc_auc_score(test[label_name], test["pred"]),roc_auc_score(valid[label_name], valid["pred"])
    print("train auc: {}, test auc: {}, valid auc: {}".format(train_auc, test_auc, valid_auc))
    importanceDf = pd.DataFrame()
    importanceDf["feature"] = pd.Series(select_features)
    importanceDf["importance_gain"] = clf.feature_importance(importance_type="gain")
    importanceDf["importance_split"] = clf.feature_importance(importance_type="split")
    importanceDf.sort_values(by="importance_gain", ascending=False, inplace=True, ignore_index=True)
    print(importanceDf.head(30))
    importanceDf.to_csv("./feature_importance.csv", index=False)

    print_rank_lift(valid["pred"], valid[label_name], percentile=0.1)



if __name__ == "__main__":

    ignore_cols = ["xiaomi_id", "credit_time", "label", "basicinfo_provider", "basicinfo_group_provider", "repay_day", "pay_first_date", "cal_ovd_date",
                     "previous_day", "join_date", "date", "effective_time", "effective_day", "date_diff", "new_bairong_tag", "pay_cash_first_date", "pay_first_month",
                     "basicinfo_total_amount", "user_rate", "pay_cash_balance_p", "prin", "pay_cash_first_amount", "sum_over_balance", "alf_apirisk_time_slope_d30_sum",
                     "mi_user_model_youpin_v2"]

    data = pd.read_csv("~/data/mi_data/data.csv")

    select_features = [col for col in data.columns if col not in ignore_cols]

    train_test_valid = get_train_test_valid(data, valid_date_bound=20200501, date_name="pay_first_date", seed=123)

    run_simple_lgb(train_test_valid, select_features)
