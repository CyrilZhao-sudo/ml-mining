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


