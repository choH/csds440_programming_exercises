'''
Author: Mingyang Tie, mxt497 Shaochen (Henry) ZHONG, sxz517
Date: 2020-10-05 10:15:45
LastEditTime: 2020-10-07 23:11:35
'''
import sys
from collections import Counter
from math import log
from random import seed

#from sklearn.metrics import roc_auc_score

from util import *


seed(12345)

def _pred(x, pre_p, posi_p, nega_p):
    """
    :return prediction of x,{0，1}
    """
    res = [log(pre_p[0]), log(pre_p[1])]  # negative / positive
    for i, attr in enumerate(x):
        res[1] += log(posi_p[i][attr])
        res[0] += log(nega_p[i][attr])

    return res


def pred(X, pre_p, posi_p, nega_p):
    ret = []
    for x in X:
        ret.append(_pred(x, pre_p, posi_p, nega_p))

    return np.asarray(ret)


def train_bayes(X, y, argv, posi_num, nega_num):
    m = len(y)  # sample number
    n = len(X[0])  # attribute number

    class_num = Counter(y.reshape(-1))
    pre_p = {}
    for k, v in class_num.items():
        pre_p[k] = v / m
    # print(pre_p)  # prior prob

    posi_p = [{} for i in range(n)]
    nega_p = [{} for i in range(n)]

    for i, d in enumerate(posi_num):
        _posi_num_i = Counter(X[y.reshape(-1) == 1, i])
        for attr in _posi_num_i:
            posi_num[i][attr] = _posi_num_i[attr]

    for i, d in enumerate(nega_num):
        _nega_num_i = Counter(X[y.reshape(-1) == 0, i])
        for attr in _nega_num_i:
            nega_num[i][attr] = _nega_num_i[attr]

    # print(posi_num)
    # m-etimate
    op4 = float(argv[3])
    if op4 < 0:
        Laplace = True
    else:
        Laplace = False

    for i, d in enumerate(posi_p):
        num_value_in_attr = len(posi_num[i])
        p = 1 / num_value_in_attr
        if Laplace:
            m = num_value_in_attr
        else:
            m = op4
        for attr in posi_num[i]:
            d[attr] = (posi_num[i][attr] + m * p) / (sum(y == 1) + m)

    for i, d in enumerate(nega_p):
        num_value_in_attr = len(nega_num[i])
        p = 1 / num_value_in_attr
        if Laplace:
            m = num_value_in_attr
        else:
            m = op4
        for attr in nega_num[i]:
            d[attr] = (nega_num[i][attr] + m * p) / (sum(y == 0) + m)

    return pre_p, posi_p, nega_p


def main():
    argv = sys.argv[1:]
    path = argv[0]
    n_bin = int(argv[2])
    X, y = read_data(path, n_bin=n_bin)

    # print(X[0])
    # print(y)
    n = len(X[0])  # attribute number
    posi_num = [{} for i in range(n)]
    nega_num = [{} for i in range(n)]

    for i, d in enumerate(posi_num):
        for attr in np.unique(X[:, i]):
            posi_num[i][attr] = 0

    for i, d in enumerate(nega_num):
        for attr in np.unique(X[:, i]):
            nega_num[i][attr] = 0
    # print(nega_p)

    cross_validation = argv[1] == '0'
    if cross_validation:
        folds = n_fold(len(X))
        AUC_y = []
        pred_AUC_y = []
        pred_AUC_y_nega = []
        # save for avg
        acc = []
        prec = []
        rec = []
        for test_fold_idx in range(5):
            # for AUC calculation

            train_idx = []
            for f in range(5):
                if f != test_fold_idx:
                    train_idx.extend(folds[f])

            trainX = X[train_idx, :]
            trainy = y[train_idx]
            testX = X[folds[test_fold_idx], :]
            testy = y[folds[test_fold_idx]]

            pre_p, posi_p, nega_p = train_bayes(trainX, trainy, argv, posi_num, nega_num)
            pred_res = pred(testX, pre_p, posi_p, nega_p)
            AUC_y.extend(testy)
            pred_AUC_y.extend(np.exp(pred_res[:, 1]))
            pred_AUC_y_nega.extend(np.exp(pred_res[:, 0]))


            _acc, _prec, _rec = cal_bayes_APR(pred_res, testy)
            report_cross(_acc,_prec,_rec)
            acc.append(_acc)
            prec.append(_prec)
            rec.append(_rec)
        roc_score = cal_AUC(AUC_y, pred_AUC_y)
        pred_AUC_y = np.asarray(pred_AUC_y)
        pred_AUC_y_nega = np.asarray(pred_AUC_y_nega)
        # print(sum((pred_AUC_y > pred_AUC_y_nega) == AUC_y))
        report(acc, prec, rec, roc_score)
    else:

        pre_p, posi_p, nega_p = train_bayes(X, y, argv, posi_num, nega_num)
        pred_res = pred(X, pre_p, posi_p, nega_p)

        roc_score = cal_AUC(y, np.exp(pred_res[:, 1]))

        acc, prec, rec = cal_bayes_APR(pred_res, y)
        report(acc, prec, rec, roc_score)

    # print("{:.03f} {:.03f}".format(np.mean(acc),np.std(acc)))




if __name__ == '__main__':
    main()
