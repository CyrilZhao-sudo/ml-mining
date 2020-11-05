# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/11/5

import pandas as pd
import numpy as np
from xmml.tools.feature_discretize import Discretize


if __name__ == "__main__":

    # data = pd.read_csv("/home/mi/data/mi_data/data.csv", usecols=["xiaomi_id", "pay_first_date", "pay_cash_first_duration",
    #                                                           "risk_score", "mi_user_model_sms_v2",
    #                                                           "mi_user_model_account",
    #                                                           "loan_amount_ratio", "label"])

    data = pd.read_csv("/Users/hudoudou/zhaochen/data/data.csv", usecols=["xiaomi_id", "pay_first_date", "pay_cash_first_duration",
                                                              "risk_score", "mi_user_model_sms_v2",
                                                              "mi_user_model_account",
                                                              "loan_amount_ratio", "label"])


    d = Discretize(df=data, cons_features=["risk_score", "mi_user_model_sms_v2", "mi_user_model_account"],
                   cate_features=None, na_val=-1)

    cuts = d.bin_freq_fit(qnt_num=10,contain_bound=False)

    # d.bin_tree_fit(max_depth=2, min_samples_leaf=0.1)

    data["d_risk_score"] = pd.cut(data["risk_score"], [-np.inf, -1, 37, 47, np.inf])
    data["d_mi_user_model_sms_v2"] = pd.cut(data["mi_user_model_sms_v2"], [-np.inf, -1.,  0.00514535,  0.0074918 ,  0.00972521,  0.01128606,
            0.01384575,  0.01585426,  0.01947729,  0.03666468,  0.13386546, np.inf])

    df_summary = d.df_summary(data, "d_mi_user_model_sms_v2", "label", is_factor=True)

    min_freq = df_summary["freq"].min()

    df,tmp =  d.merge_bin_freq(data, df_summary, "d_mi_user_model_sms_v2", min_freq, threshold=0.05, is_factor=True)