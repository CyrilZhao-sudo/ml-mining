# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/11/5

import pandas as pd
import numpy as np
from xmml.tools.feature_discretize import Discretize



data = pd.read_csv("/home/mi/data/mi_data/data.csv", usecols=["xiaomi_id", "pay_first_date", "pay_cash_first_duration",
                                                              "risk_score", "mi_user_model_sms_v2",
                                                              "mi_user_model_account",
                                                              "loan_amount_ratio", "label"])
d = Discretize(df=data, cons_features=["risk_score", "mi_user_model_sms_v2", "mi_user_model_account"],
               cate_features=None, na_val=-1)

cuts = d.bin_freq_fit(qnt_num=3,contain_bound=False)

d.bin_tree_fit(max_depth=2, min_samples_leaf=0.1)

data["d_risk_score"] = pd.cut(data["risk_score"], [-np.inf, -1, 37, 47, np.inf])

Interval(47.0, inf, closed='right')