'''
Author: Mingyang Tie, mxt497 Shaochen (Henry) ZHONG, sxz517
Date: 2020-10-05 12:38:42
LastEditTime: 2020-10-09 11:09:31
'''
from random import choice

import numpy as np

from preprocess import process
from mldata import parse_c45
import matplotlib.pyplot as plt


def report_cross(acc, prec, rec):
    print("===============Fold report==================")
    print('Accuracy:{:.03f}'.format(acc))
    print('Precision:{:.03f}'.format(prec))
    print('Recall:{:.03f}'.format(rec))


def report(acc, prec, rec, auc):
    print("===============Final report=================")
    if type(acc) is list:
        print('Accuracy:{:.03f} {:.03f}'.format(np.mean(acc), np.std(acc)))
        print('Precision:{:.03f} {:.03f}'.format(np.mean(prec), np.std(prec)))
        print('Recall:{:.03f} {:.03f}'.format(np.mean(rec), np.std(rec)))
        print('Area under ROC {:.03f}'.format(auc))
    else:
        print('Accuracy:{:.03f} {:.03f}'.format(acc, 0))
        print('Precision:{:.03f} {:.03f}'.format(prec, 0))
        print('Recall:{:.03f} {:.03f}'.format(rec, 0))
        print('Area under ROC {:.03f}'.format(auc))


def n_fold(n_sample, n_fold=5):
    a = list(range(n_sample))
    folds = [[] for i in range(n_fold)]
    fold_ptr = 0
    while len(a) > 0:
        t = choice(a)
        a.remove(t)
        folds[fold_ptr].append(t)
        fold_ptr = (fold_ptr + 1) % n_fold

    return folds


def read_data(path, n_bin=3):
    prob_name = path.split('/')[-1]
    datafile = path + '/' + prob_name + '.data'
    # data = np.loadtxt(datafile, delimiter=',', dtype=str)
    data = parse_c45(prob_name, path)
    data = np.asarray(data.to_float())
    # print(data)

    X = data[:, 1:-1]
    X = process(X, prob_name, n_bin)
    y = data[:, -1].astype(int)
    return X, y


def compute_tp_tn_fn_fp(y, y_hat):
    tp = sum((y == 1) & (y_hat == 1))
    tn = sum((y == 0) & (y_hat == 0))
    fn = sum((y == 1) & (y_hat == 0))
    fp = sum((y == 0) & (y_hat == 1))
    return tp, tn, fp, fn


def compute_accuracy(tp, tn, fn, fp):
    return (tp + tn) / float(tp + tn + fn + fp)


def compute_precision(tp, fp):
    return tp / float(tp + fp)


def compute_recall(tp, fn):
    return tp / float(tp + fn)


def cal_APR(y_hat, y):
    tp, tn, fp, fn = compute_tp_tn_fn_fp(y, y_hat)

    acc = compute_accuracy(tp, tn, fn, fp)
    prec = compute_precision(tp, fp)
    rec = compute_recall(tp, fn)

    return acc, prec, rec


def cal_LR_APR(pred, y):
    n = len(y)
    y_hat = pred > 0.5
    # y_hat=np.asarray(y_hat)
    # y_hat=y_hat.reshape(-1)
    return cal_APR(y_hat, y)


def cal_bayes_APR(pred_res, y):
    """
    accuracy,precision,recall
    :param pred_res m*2
    :return:
    """
    n = len(y)
    nega = pred_res[:, 0]
    posi = pred_res[:, 1]
    y_hat = posi > nega
    # y_hat=[1 if ]
    return cal_APR(y_hat, y)


def cal_AUC(y, y_hat, num_bins=10000):
    postive_len = sum(y)
    negative_len = len(y) - postive_len
    total_grid = postive_len * negative_len
    pos_histogram = [0 for _ in range(num_bins + 1)]
    neg_histogram = [0 for _ in range(num_bins + 1)]
    bin_width = 1.0 / num_bins
    for i in range(len(y)):
        nth_bin = int(y_hat[i] / bin_width)
        # print(nth_bin)
        pos_histogram[nth_bin] += 1 if y[i] == 1 else 0
        neg_histogram[nth_bin] += 1 if y[i] == 0 else 0
    accu_neg = 0
    satisfied_pair = 0
    for i in range(num_bins):
        satisfied_pair += pos_histogram[i] * accu_neg + pos_histogram[i] * neg_histogram[i] * 0.5
        accu_neg += neg_histogram[i]

    return satisfied_pair / float(total_grid)


def plot_roc(y_true, y_score):
    y_true=np.asarray(y_true)
    y_score=np.asarray(y_score)
    pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    out = np.cumsum(y_true, dtype=np.float64)
    expected = np.sum(y_true, dtype=np.float64)
    tps = out[threshold_idxs]

    fps = 1 + threshold_idxs - tps
    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    fpr = fps / fps[-1]

    tpr = tps / tps[-1]
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', )
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


if __name__ == "__main__":
    plot_roc(1,1)