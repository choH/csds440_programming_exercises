
import sys
from collections import Counter
from math import log
import random
from sklearn.metrics import roc_auc_score
from util import *
import math

random.seed(12345)

def _pred(x, pre_p, posi_p, nega_p):
    """
    :return prediction of x,{0，1}
    """
    res = [log(pre_p[0]), log(pre_p[1])]  # negative / positive
    for i, attr in enumerate(x):
        res[1] += log(posi_p[i][attr])
        res[0] += log(nega_p[i][attr])

    if res[1] > res[0]:
        res[1] = 1
        res[0] = 0
    else:
        res[1] = 0
        res[0] = 1
    return res


def pred(X, pre_p, posi_p, nega_p):
    ret = []
    for x in X:
        ret.append(_pred(x, pre_p, posi_p, nega_p))

    return np.asarray(ret)

def boosttrain_bayes(x_train, y_train, m_etimate, posi_num, nega_num, wboost, epsilon_thread):
    m = len(y_train)  # sample number
    n = len(x_train[0])  # attribute number

    class_num = Counter(y_train.reshape(-1))
    pre_p = {}
    for k, v in class_num.items():
        pre_p[k] = v / m

    posi_p = [{} for i in range(n)]
    nega_p = [{} for i in range(n)]
    for i, d in enumerate(posi_num): # Weighted Data
        _posi_num_i = {}
        count = x_train[:, i]
        _posi_num_i_count = Counter(x_train[y_train.reshape(-1) == 1, i])
        for j in range(len(y_train)):
            if y_train[j]==0:
                continue
            if count[j] in _posi_num_i:
                _posi_num_i[count[j]] = _posi_num_i[count[j]] + wboost[j][0]
            else:
                _posi_num_i[count[j]] = wboost[j][0]
        attr_sum = 0.0
        for attr in _posi_num_i:
            attr_sum = attr_sum + _posi_num_i[attr]
        for attr in _posi_num_i:
            x_tmp = _posi_num_i[attr]/attr_sum
            posi_num[i][attr] = int(x_tmp*_posi_num_i_count[attr])
            
    for i, d in enumerate(nega_num):
        _nega_num_i = {}
        count = x_train[:, i]
        _nega_num_i_count = Counter(x_train[y_train.reshape(-1) == 0, i])
        for j in range(len(y_train)):
            if y_train[j]==1:
                continue
            if count[j] in _nega_num_i:
                _nega_num_i[count[j]] = _nega_num_i[count[j]] + wboost[j][0]
            else:
                _nega_num_i[count[j]] = wboost[j][0]
        attr_sum = 0.0
        for attr in _nega_num_i:
            attr_sum = attr_sum + _nega_num_i[attr]
        for attr in _nega_num_i:
            x_tmp = _nega_num_i[attr]/attr_sum
            nega_num[i][attr] = int(x_tmp*_nega_num_i_count[attr])

        # _nega_num_i = Counter(x_train[y_train.reshape(-1) == 0, i])
        # for attr in _nega_num_i:
        #     nega_num[i][attr] = _nega_num_i[attr]

    # m-etimate
    op4 = m_etimate
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
            d[attr] = (posi_num[i][attr] + m * p) / (sum(y_train == 1) + m)
    
    for i, d in enumerate(nega_p):
        num_value_in_attr = len(nega_num[i])
        p = 1 / num_value_in_attr
        if Laplace:
            m = num_value_in_attr
        else:
            m = op4
        for attr in nega_num[i]:
            d[attr] = (nega_num[i][attr] + m * p) / (sum(y_train == 0) + m)
    
    y_predB = pred(x_train, pre_p, posi_p, nega_p)
    y_pred = []
    for i in y_predB:
        if i[0] > i[1]:
            y_pred.append(0)
        else:
            y_pred.append(1)
    y_pred = np.array(y_pred)
    epsilon = wboost.transpose().dot(y_train^ y_pred)
    # print(epsilon)
    if epsilon <= epsilon_thread or epsilon >= 0.5:
        return pre_p, posi_p, nega_p, epsilon, 0, wboost
    alpha = 0.5 * math.log((1-epsilon)/epsilon)
    wboost_ = wboost
    exp_ = np.exp(-alpha*(y_train^ y_pred))
    wboost_ = wboost_.reshape(len(wboost_), 1) * exp_.reshape(len(wboost_), 1)
    wboost_ = wboost_/np.sum(wboost_)
    return pre_p, posi_p, nega_p, epsilon, alpha, wboost_


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

        #cancel comment if you want to plot the ROC
        # plot_roc(y,np.exp(pred_res[:,1]))

        roc_score = cal_AUC(y, np.exp(pred_res[:, 1]))

        acc, prec, rec = cal_bayes_APR(pred_res, y)
        report(acc, prec, rec, roc_score)

    # print("{:.03f} {:.03f}".format(np.mean(acc),np.std(acc)))



if __name__ == '__main__':
    main()
