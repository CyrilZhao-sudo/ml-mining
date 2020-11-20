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

    # data = pd.read_csv("/Users/hudoudou/zhaochen/data/data.csv", usecols=["xiaomi_id", "pay_first_date", "pay_cash_first_duration",
    #                                                           "risk_score", "mi_user_model_sms_v2",
    #                                                           "mi_user_model_account",
    #                                                           "loan_amount_ratio", "label"])


    # d = Discretize(train_df=data, cons_features=["risk_score", "mi_user_model_sms_v2", "mi_user_model_account"],
    #                cate_features=None, na_val=-1)
    #
    # d.bin_freq_fit(qnt_num=10,contain_bound=False)
    # print(d.cons_bins)
    # print(d.iv_df)
    # d.bin_tree_fit(max_depth=3, min_samples_leaf=30)
    # print(d.cons_bins)
    # print(d.iv_df)
    #
    # chi_cuts = d.bin_chi_single(df=data, feat_name="risk_score", is_continuous=True, threshold=0.05, p_val=0.05, init_freq_bins=6)
    # print(chi_cuts)
    # print(d.woe_transform())


    # data["d_risk_score"] = pd.cut(data["risk_score"], [-np.inf, -1, 37, 47, np.inf])
    # data["d_mi_user_model_sms_v2"] = pd.cut(data["mi_user_model_sms_v2"], [-np.inf, -1.,  0.00514535,  0.0074918 ,  0.00972521,  0.01128606,
    #         0.01384575,  0.01585426,  0.01947729,  0.03666468,  0.13386546, np.inf])
    # print(d.onehot_transform())
    # data["d_mi_user_model_sms_v2"].cat.rename_categories({pd.Interval(-np.inf, -1.0):pd.Interval(-np.inf, 0.00515),
    #                                                       pd.Interval(-1.0, 0.00515):pd.Interval(-np.inf, 0.00515)})
    # data["d_mi_user_model_sms_v2"].cat.remove_categories([pd.Interval(-np.inf, -1.0), pd.Interval(-1.0, 0.00515)]).cat.add_categories([pd.Interval(-np.inf, 0.00515)]).fillna(pd.Interval(-np.inf, 0.00515))
    #
    # t = data["d_mi_user_model_sms_v2"].cat.remove_categories([pd.Interval(-np.inf, -1.0), pd.Interval(-1.0, 0.00515)]).cat.add_categories([pd.Interval(-np.inf, 0.00515)]).fillna(pd.Interval(-np.inf, 0.00515))
    # sorted(t.cat.categories.to_list())
    # s_idx =sorted(t.cat.categories.to_list())
    # t.cat.reorder_categories(s_idx)

    # df_summary = d.df_summary(data, "d_mi_user_model_sms_v2", "label", is_factor=True)
    #
    # min_freq = df_summary["freq"].min()
    #
    # df_xy, tmp =  d.merge_bin_freq(data[["d_mi_user_model_sms_v2", "label"]], df_summary, "d_mi_user_model_sms_v2", min_freq, threshold=0.05, is_factor=True)


    def replace_category_values(x, ori_values, new_value):
        # 处理category的数据
        x_new = x.cat.remove_categories(ori_values).cat.add_categories(new_value).fillna(new_value)
        new_categories = sorted(x_new.cat.categories.to_list())
        return x_new.cat.reorder_categories(new_categories, ordered=True)


    # train = pd.read_csv("/home/mi/PycharmProjects/ml-mining/data/credit/train.csv")
    # # ["年龄", "商品总价", "首付金额", "贷款金额"]
    # d = Discretize(train, cons_features=[],
    #                cate_features=["受教育程度","是否保险","是否随心包","住房类型"], label_identify="OVDU", cache_stat=True)
    #
    # d.bin_freq_fit(qnt_num=5)
    # print(d.iv_df)
    # print(d.cate_bins)
    # print(d.woe_stats["受教育程度"])
    # for k, v in d.woe_stats.items():
    #     print(v)
    #
    # test = pd.read_csv("/home/mi/PycharmProjects/ml-mining/data/credit/test.csv")
    # test.loc[0, "受教育程度"] = 999
    # test_woe = d.woe_transform(test)
    # print(test_woe.head())
    # print(d.woe_transform().head())

    fal_train = pd.read_excel("/home/mi/data/com/fal/fal_data_train.xlsx", index_col=0)
    # fal_train[["FPD10", "FPD30", "FPD60", "FPD90", "SPD10", "SPD30", "TPD10", "TPD30"]].corr()
    # fal_train[["FPD30", "FPD90", "SPD30", "TPD30"]].corr()
    fal_train["ovd"] = fal_train["FPD30"] + fal_train["SPD30"] + fal_train["TPD30"]
    fal_train.fillna(-1, inplace=True)
    acm_cols = [acm for acm in fal_train.columns if "ACM_" in acm]
    al_cols = [al for al in fal_train.columns if "AL_" in al]
    cate_cols = ["EMPLOYRECORD_NAME","CELLPROPERTY_NAME","EDUEXPERIENCE", "EFFECTIVEANNUALRATE", "EMAIL_TYPE",
                 "MARRIGE", "GOODS_CATEGORY", "PRODUCTCTYPENAME"]
    acm_cols + al_cols + ["max_type_al"]
    d = Discretize(fal_train, cons_features= ["ACM_M1_DEBIT_REPAY"],
                   cate_features=[], label_identify="ovd", cache_stat=True, na_val=-1)

    d.bin_chi_fit(threshold=0.05, p_val=0.05, init_freq_bins=10)


    # d.bin_tree_fit(max_depth=3, min_samples_leaf=10)
    #
    # d.iv_df.to_csv("iv.csv", index=False)
    #
    # select_cols = d.iv_df.sort_values(by=["iv"], ascending=False).loc[d.iv_df["iv"]>=0.02, "feature"].to_list()
    #
    # corr = fal_train[select_cols].corr()
    # corr.to_csv("corr.csv")