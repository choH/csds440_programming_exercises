'''
Author: Mingyang Tie, mxt497 Shaochen (Henry) ZHONG, sxz517
Date: 2020-10-05 12:38:42
LastEditTime: 2020-10-09 11:09:31
'''
from random import choice

import numpy as np

from preprocess import process
from mldata import parse_c45

def report_cross(acc, prec, rec):
    print("===============Fold report==================")
    print('Accuracy:{:.03f}'.format(acc))
    print('Precision:{:.03f}'.format(prec))
    print('Recall:{:.03f}'.format(rec))
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



def compute_accuracy(tp, tn, fn, fp):
    return (tp + tn) / float(tp + tn + fn + fp)


def compute_precision(tp, fp):
    return tp / float(tp + fp)


def compute_recall(tp, fn):
    return tp / float(tp + fn)


def cal_APR(y_hat, y):
    tp, tn, fp, fn = compute_tp_tn_fn_fp(y, y_hat)
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
