# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/10/14

import pandas as pd
import numpy as np
from collections import Counter,OrderedDict
import logging
import time
from sklearn.compose import ColumnTransformer,
from sklearn.impute import SimpleImputer

def get_logger(logger_name=None):
    if logger_name:
        return logging.getLogger(f"TableauDeep - {logger_name}")
    else:
        return logging.getLogger(f"TableauDeep")


logger = get_logger()


class DataFrameWrapper:
    def __init__(self, transform, columns=None):
        self.transformer = transform
        self.columns = columns

    def fit(self, X):
        if self.columns is None:
            self.columns = X.columns.tolist()
        self.transformer.fit(X)
        return self

    def transform(self, X):
        df = pd.DataFrame(self.transformer.transform(X))
        df.columns = self.columns
        return df

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class DataProcess:
    def __init__(self, cons_cols, cate_cols, cons_process_type='norm', cate_process_type='encode'):
        self.cons_cols = cons_cols
        self.cate_cols = cate_cols
        self.cons_process_type = cons_process_type
        self.cate_process_type = cate_process_type

        self.X_transformers = OrderedDict()

    def prepare_X(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        if len(set(X.columns)) != len(list(X.columns)):
            cols = [item for item, count in Counter(X.columns).items() if count > 1]
            raise ValueError(f'Columns with duplicate names in X: {cols}')
        if X.columns.dtype != 'object':
            X.columns = ['x_' + str(c) for c in X.columns]
            logger.warn(f"Column index of X has been converted: {X.columns}")
        return X

    def fit_transform(self, X, y, copy_data=True):

        y_df = pd.DataFrame(y)
        if y_df.isnull().sum().sum() > 0:
            raise ValueError("Missing values in y.")

        if copy_data:
            import copy
            X = copy.deepcopy(X)
            y = copy.deepcopy(y)

        #y = self.fit_transform_y(y)

        X = self.prepare_X(X)
        if self.cons_process_type == 'norm':
            # 标准化
            pass
        else:
            # 分箱
            pass

        if self.cate_process_type == "encode":
            pass
        else:
            pass





    def transform(self, X_df, y):
        pass

    def fit_transform_y(self, y):
        pass


    def _imputation(self, X):
        start = time.time()
        logger.info('Data imputation...')
        continuous_vars = self.cons_cols
        categorical_vars = self.cate_cols
        ct = ColumnTransformer([
            ('categorical', SimpleImputer(missing_values=np.nan, strategy='constant'),
             categorical_vars),
            ('continuous', SimpleImputer(missing_values=np.nan, strategy='mean'), continuous_vars),
        ])
        dfwrapper = DataFrameWrapper(ct, categorical_vars + continuous_vars)
        X = dfwrapper.fit_transform(X)
        self.X_transformers['imputation'] = dfwrapper
        print(f'Imputation cost:{time.time() - start}')
        return X


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    cols_renames = {'年龄': 'age', '性别': 'sex', '婚姻状况': 'married', '同盾': 'tongdun', '平台': 'platform', '商品总价': 'price',
                    '首付金额': 'first_pay_amount',
                    '贷款金额': 'loan_amount', '分期期数': 'install_num', '是否保险': 'is_insurance', '是否随心包': 'is_heart',
                    '住房类型': 'house_type',
                    '手机使用时长': 'phone_used_time', '是否实名': 'is_true_name', '受教育程度': 'edu', '是否办理过分期业务': 'is_install',
                    '月收入': 'income', '芝麻信用分': 'zm_score',
                    'OVDU': 'label'}

    train_raw = pd.read_csv("./data/credit/train.csv")
    train_raw.rename(columns=cols_renames, inplace=True)
    test_raw = pd.read_csv("./data/credit/test.csv")
    test_raw.rename(columns=cols_renames, inplace=True)
    train, valid = train_test_split(train_raw, test_size=0.2)
