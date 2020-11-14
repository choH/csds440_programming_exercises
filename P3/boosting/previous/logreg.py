
import sys
import random
import math
from util import *

random.seed(12345)

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

def boostLR(x_train, y_train, wboost, epsilon_thread, max_iters=500, lbd=0.1):
    learning_rate = 0.001  # step
    X = np.mat(x_train)
    y = np.mat(y_train).transpose()
    m, n = np.shape(X)
    # init
    weights = np.ones((n, 1))

    for i in range(max_iters):
        y_hat = sigmoid(X * weights)  # sigmoid
        grad = X.transpose() * (y - y_hat) + lbd * weights 
        weights = weights + learning_rate * grad  # grad
    
    y_pred = pred(X, weights)
    y_pred[y_pred<0.5] = 0
    y_pred[y_pred>=0.5] = 1
    y_pred = y_pred.astype(int)
    epsilon = wboost.transpose().dot(y_train^ y_pred)
    # print(epsilon)
    if epsilon <= epsilon_thread or epsilon >= 0.5:
        return weights, epsilon, 0, wboost
    alpha = 0.5 * math.log((1-epsilon)/epsilon)
    wboost_ = wboost
    exp_ = np.exp(-alpha*(y_train^ y_pred))
    wboost_ = wboost_.reshape(len(wboost_), 1) * exp_.reshape(len(wboost_), 1)
    wboost_ = wboost_/np.sum(wboost_)
    return weights, epsilon, alpha, wboost_

def pred(X, weight):
    return np.asarray(sigmoid(X * weight)).reshape(-1)

# logreg.py ./440data/spam 0 0.1

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

        #cancel comment if you want to plot the ROC
        #plot_roc(y,pred_res)
        acc, prec, rec = cal_LR_APR(pred_res, y)
        roc_score = cal_AUC(y, pred_res)
        report(acc, prec, rec, roc_score)


if __name__ == '__main__':
    main()

# def boostpred(x_test, weights, wboost):
#     y_hat = sigmoid(x_test * weights)
#     return x_test
#     # h_x = np.asarray(sigmoid(x_test * weight)).reshape(-1)