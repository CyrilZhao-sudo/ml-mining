# -*- coding: utf-8 -*-
# Author: zhao chen
# Date: 2020/8/29

import numpy as np
import math
import pandas as pd

np.arange(0.0, 1.1, 0.1)

def getKsValue(y_pred, y_true, weight=None, threshold=0.5):
    if not weight:
        weight = np.ones_like(y_true)
    items = [(y_pred[i], y_true[i], weight[i]) for i in range(len(y_pred))]
    itemsSorted = sorted(items, key=lambda x: x[0], reverse=True)
    goodCumCount, badCumCount, goodCount, badCount = 0, 0, len(y_true) - sum(y_true), sum(y_true)
    ksValue = 0
    for (p, t, w) in itemsSorted:
        if t > threshold:
            badCumCount += w
        else:
            goodCumCount += w
        ks = math.fabs(badCumCount / badCount - goodCumCount / goodCount)
        ksValue = ks if ks > ksValue else ksValue
    return ksValue


def getPsiValue(base_pred, observe_pred, bins=10):
    data = pd.DataFrame({"base": base_pred, "observe": observe_pred})
    arrayBin, cutPoints = pd.qcut(data["base"], q=bins, retbins=True, duplicates="drop")
    data["baseBin"] = arrayBin



def tpr_weight_function(y_true, y_predict):
    df = pd.DataFrame()
    df['prob'] = list(y_predict)
    df['y'] = list(y_true)
    df = df.sort_values(by=['prob'], ascending=False, ignore_index=True)
    y = df.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = df['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3

