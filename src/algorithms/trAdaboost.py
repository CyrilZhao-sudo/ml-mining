# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/8/27

import numpy as np
import pandas as pd
from math import sqrt, log1p
from sklearn.metrics import roc_auc_score


class TrAdaBoost(object):

    def __init__(self, learner):
        self.learner = learner
        self.best_clf = None
        self.clf_cache = None
        self.p_cache = None
        self.beta = None

    def fit(self, old_train, new_train, num_rounds, valid=None, init_weights=(1.0, 1.0)):

        n, m = len(old_train), len(new_train)
        beta = 1 / (1 + sqrt(2 * log1p(n / num_rounds)))
        w1, w2 = init_weights
        old_w = np.ones(n) / n * w1
        new_w = np.ones(m) / m * w2
        weights = np.concatenate([old_w, new_w], axis=0)
        train = pd.concat([old_train, new_train], axis=0).drop(columns=["no"])
        prob_cache = np.zeros((m, num_rounds))
        p_cache = np.zeros((n + m, num_rounds))
        clf_cache = []
        best_score = 0

        for t in range(num_rounds):
            print("<- {} round ->".format(t))
            p_t = self.calculate_p(weights)
            p_cache[:, t] = p_t

            self.learner.fit(X=train.drop(columns=["label"]), y=train["label"], sample_weight=p_t.reshape(-1, 1) * 100)
            prob = self.learner.predict_proba(train.drop(columns=["label"]))[:, 1]
            auc_train = roc_auc_score(y_true=train["label"], y_score=prob)
            auc_new = roc_auc_score(y_true=new_train["label"], y_score=prob[n:])
            print("train data auc:{0} , new data auc:{1}".format(auc_train, auc_new))

            if valid:
                prob_valid = self.learner.predict_proba(valid.drop(columns=["label"]))[:, 1]
                auc_valid = roc_auc_score(y_true=valid["label"], y_score=prob_valid)
                print("valid data auc:{}".format(auc_new, auc_valid))

            clf_cache.append(self.learner)

            e_t = self.calculate_e(new_train["label"].values, prob[n:], p_t[n:])
            if e_t > 0.5:
                e_t = 0.5
            if e_t == 0.:
                print("Round {} , early stopping.".format(t + 1))
                break

            beta_t = e_t * (1 - e_t)
            weights[:n] = p_t[:n] * (beta ** np.abs(old_train["label"].values - prob[:n]))
            weights[n:] = p_t[n:] * (beta_t ** (-np.abs(new_train["label"].values - prob[n:])))
            prob_cache[:, t] = prob[n:]

            # print("beta {0}. beta_t {1}".format(beta, beta_t))

            if auc_new > best_score:
                best_score = auc_new
                best_round = t

        self.clf_cache = clf_cache[:]
        self.p_cache = p_cache[:]
        self.prob_cache = prob_cache
        self.best_clf = clf_cache[best_round]
        self.beta = beta
        self.num_rounds = num_rounds

    def predict(self, data):
        pred = self.best_clf.predict_proba(data)[:, 1]
        return pred

    def blend_predict(self, data, n=0):
        pred = np.zeros(len(data))
        for i in range(n, self.num_rounds):
            pred += self.clf_cache[i].predict_proba(data)[:, 1]

        return pred / (self.num_rounds - n)

    @staticmethod
    def calculate_p(w):
        if not isinstance(w, np.ndarray):
            w = np.asarray(w)
        return w / np.sum(w)

    @staticmethod
    def calculate_e(ht, c, wt):
        return np.sum(np.abs(ht - c) * wt / np.sum(wt))


if __name__ == "__main__":
    dir_path = "/home/mi/PycharmProjects/ml-mining/data/DataTrain"
    old_train = pd.read_csv(dir_path + "/A_train.csv")
    new_train = pd.read_csv(dir_path + "/B_train.csv")
    test = pd.read_csv(dir_path + "/B_test.csv")
    old_train.rename(columns={"flag": "label"}, inplace=True)
    new_train.rename(columns={"flag": "label"}, inplace=True)

    from xgboost import XGBClassifier

    learner = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=7, subsample=0.8, colsample_bytree=0.8,
                            objective="binary:logistic", reg_lambda=1, reg_alpha=1, random_state=123, booster="gbtree",
                            n_jobs=-1, verbosity=1)
    # learner.fit()

    trAdaboost = TrAdaBoost(learner=learner)

    trAdaboost.fit(old_train=old_train, new_train=new_train, num_rounds=10)

    pred = trAdaboost.predict(test.drop(columns=["no"]))

    pred = trAdaboost.blend_predict(test.drop(columns=["no"]), n=2)

    sub = test[["no"]]
    sub["pred"] = pred

    sub.to_csv("blend_sub.csv", index=False)
