'''
Author: Mingyang Tie, mxt497 Shaochen (Henry) ZHONG, sxz517
Date: 2020-10-07 12:08:24
LastEditTime: 2020-10-09 14:44:35
'''
import sys
from random import seed

from util import *

seed(12345)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def LR(X, y, lbd=0.1):
    learning_rate = 0.001  # step

    X = np.mat(X)
    y = np.mat(y).transpose()
    m, n = np.shape(X)
    # init
    weights = np.ones((n, 1))
    max_iters = 500  # max loop

    for i in range(max_iters):
        y_hat = sigmoid(X * weights)  # sigmoid
        grad = X.transpose() * (y - y_hat) + lbd * weights
        weights = weights + learning_rate * grad  # grad
    return weights


def pred(X, weight):
    return np.asarray(sigmoid(X * weight)).reshape(-1)


def main():
    argv = sys.argv[1:]

    path = argv[0]
    X, y = read_data(path)
    lbd = float(argv[2])
    cross_validation = argv[1] == '0'
    folds = n_fold(len(X))
    if cross_validation:
        AUC_y = []
        pred_AUC_y = []

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
            weight = LR(trainX, trainy, lbd)
            pred_res = pred(testX, weight)

            AUC_y.extend(testy)
            pred_AUC_y.extend(pred_res)

            _acc, _prec, _rec = cal_LR_APR(pred_res, testy)
            report_cross(_acc, _prec, _rec)
            acc.append(_acc)
            prec.append(_prec)
            rec.append(_rec)
        roc_score = cal_AUC(AUC_y, pred_AUC_y)
        report(acc, prec, rec, roc_score)
    else:
        weight = LR(X, y, lbd)
        pred_res = pred(X, weight)
        acc, prec, rec = cal_LR_APR(pred_res, y)
        roc_score = cal_AUC(y, pred_res)
        report(acc, prec, rec, roc_score)


if __name__ == '__main__':
    main()
