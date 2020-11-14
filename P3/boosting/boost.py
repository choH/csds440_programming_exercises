import random
import mldata
import util
import numpy as np
import preprocess
import previous.logreg as logreg
import previous.nbayes as nbayes
from previous.ID3_dtree import ID3DecisionTree 
import sys

random.seed(12345)

def create_for_train(X_data, y_data, folds, n_bin, index):
    if n_bin == 1:
        return X_data, y_data, X_data, y_data
    else:
        x_train = []
        y_train = []
        x_test = folds[index]
        y_test = folds[index]
        for i in range(n_bin):
            if i == index:
                continue
            x_train.extend(folds[i])
            y_train.extend(folds[i])

        return X_data[x_train, :], y_data[y_train], X_data[x_test, :], y_data[y_test]

def boost(path, option, solver_type, num_iters): 
    path = path.replace("\\", "/")
    file_base = path.split('/')[-1]
    rootdir=path

    epsilon_thread = 0.00000001

    data = mldata.parse_c45(file_base, rootdir)
    n_bin = 1
    cross_validation = False
    if option == 0:
        n_bin = 5
        cross_validation = True

    data = np.asarray(data.to_float())
    X_data = data[:, 1:-1]
    X_data = preprocess.process(X_data, file_base, n_bin)
    y_data = data[:, -1].astype(int)
    # print(len(X_data))
    # partition the data into multiple dataset,
    folds = util.n_fold(len(data), n_bin)

    # nbayes:
    posi_num = [{} for i in range(len(X_data[0]))]
    nega_num = [{} for i in range(len(X_data[0]))]
            
    for i, d in enumerate(posi_num):
        for attr in np.unique(X_data[:, i]):
            posi_num[i][attr] = 0
                    
    for i, d in enumerate(nega_num):
        for attr in np.unique(X_data[:, i]):
            nega_num[i][attr] = 0

    AUC_y = []
    pred_AUC_y = []
    acc = []
    prec = []
    rec = []
    # training and evaluating
                
    for i in range(n_bin):
        if solver_type == "dtree":
            tree = ID3DecisionTree(1, path, "gain", cross_validation)
            x_train, y_train, x_test, y_test = tree.create_for_train(n_bin, i)
            train_size = len(x_train)
            wboost = np.ones((train_size, 1)).astype(float)/train_size
            alphas = []
            epsilons = []
            forest = []
            for iter_ in range(num_iters): 
                tree = ID3DecisionTree(1, path, "gain", cross_validation)
                x_train, y_train, x_test, y_test = tree.create_for_train(n_bin, i)
                D_train = (x_train, y_train)
                wboost, epsilon, alpha = tree.boosttrain(D_train, wboost, epsilon_thread)
                forest.append(tree)
                epsilons.append(epsilon)
                if epsilon == 0:
                    alphas = [0] * len(alphas)
                    alphas.append(1)
                    break
                elif epsilon <= epsilon_thread or epsilon >= 0.5:
                    alphas.append(alpha)
                    break
                else:
                    alphas.append(alpha)
                # y_pred = tree.test(x_test)
            result = []
            for i in range(len(forest)):
                y_predB = forest[i].test(x_test)
                y_pred = np.array(y_predB)
                y_pred[y_pred==False] = 0
                y_pred[y_pred==True] = 1
                result.append(y_pred)
            alphas = np.array(alphas)
            alphas = alphas/np.sum(alphas)
            y_pred = alphas.dot(np.array(result))
            y_pred[y_pred<0.5] = 0
            y_pred[y_pred>=0.5] = 1
            y_test = np.array(y_test)
            y_test[y_test<0.5] = 0
            y_test[y_test>=0.5] = 1
            AUC_y.extend(y_test)
            pred_AUC_y.extend(y_pred)
            _acc, _prec, _rec = util.cal_APR(y_pred, y_test)
            if cross_validation:
                util.report_cross(_acc, _prec, _rec)
            acc.append(_acc)
            prec.append(_prec)
            rec.append(_rec)
        elif solver_type == "nbayes":
            m_etimate = 0.1
            x_train, y_train, x_test, y_test = create_for_train(X_data, y_data, folds, n_bin, i)
            train_size = len(x_train)
            wboost = np.ones((train_size, 1)).astype(float)/train_size
            alphas = []
            epsilons = []
            pre_ps = []
            posi_ps = []
            nega_ps = []
            for iter_ in range(num_iters): 
                pre_p, posi_p, nega_p, epsilon, alpha, wboost = nbayes.boosttrain_bayes(x_train, y_train, m_etimate, posi_num, nega_num, wboost, epsilon_thread)
                epsilons.append(epsilon)
                pre_ps.append(pre_p)
                posi_ps.append(posi_p)
                nega_ps.append(nega_p)
                if epsilon == 0:
                    alphas = [0] * len(alphas)
                    alphas.append(1)
                    break
                elif epsilon <= epsilon_thread or epsilon >= 0.5:
                    alphas.append(alpha)
                    break
                else:
                    alphas.append(alpha)
            result = []
            for i in range(len(pre_ps)):
                y_predB = nbayes.pred(x_test, pre_ps[i], posi_ps[i], nega_ps[i])
                y_pred = []
                for i in y_predB:
                    if i[0] > i[1]:
                        y_pred.append(0)
                    else:
                        y_pred.append(1)
                y_pred = np.array(y_pred)
                result.append(y_pred)
            alphas = np.array(alphas)
            alphas = alphas/np.sum(alphas)
            y_pred = alphas.dot(np.array(result))
            y_pred[y_pred<0.5] = 0
            y_pred[y_pred>=0.5] = 1
            AUC_y.extend(y_test)
            pred_AUC_y.extend(y_pred)
            _acc, _prec, _rec = util.cal_APR(y_pred, y_test)
            if cross_validation:
                util.report_cross(_acc, _prec, _rec)
            acc.append(_acc)
            prec.append(_prec)
            rec.append(_rec)
        elif solver_type == "logreg":
            # train
            x_train, y_train, x_test, y_test = create_for_train(X_data, y_data, folds, n_bin, i)
            train_size = len(x_train)
            wboost = np.ones((train_size, 1)).astype(float)/train_size
            alphas = []
            epsilons = []
            weights = []
            for iter_ in range(num_iters): 
                weight, epsilon, alpha, wboost = logreg.boostLR(x_train, y_train, wboost, epsilon_thread, max_iters=500, lbd=0.1)
                epsilons.append(epsilon)
                weights.append(weight)
                if epsilon == 0:
                    alphas = [0] * len(alphas)
                    alphas.append(1)
                    break
                elif epsilon <= epsilon_thread or epsilon >= 0.5:
                    alphas.append(alpha)
                    break
                else:
                    alphas.append(alpha)
            result = []
            for i in range(len(weights)):
                result.append(logreg.pred(x_test, weights[i]))
            alphas = np.array(alphas)
            alphas = alphas/np.sum(alphas)
            y_pred = alphas.dot(np.array(result))
            y_pred[y_pred<0.5] = 0
            y_pred[y_pred>=0.5] = 1
            AUC_y.extend(y_test)
            pred_AUC_y.extend(y_pred)
            _acc, _prec, _rec = logreg.cal_LR_APR(y_pred, y_test)
            if cross_validation:
                util.report_cross(_acc, _prec, _rec)
            acc.append(_acc)
            prec.append(_prec)
            rec.append(_rec)
        else:
            return 
    roc_score = logreg.cal_AUC(AUC_y, pred_AUC_y)
    util.report(acc, prec, rec, roc_score)


# call boost function
if __name__ == '__main__':
    argv = sys.argv
    boost(argv[1], int(argv[2]), argv[3], int(argv[4]))
    # boost("440data/spam", 0, "logreg", 50)
    # boost("440data/volcanoes", 0, "logreg", 50)
    # boost("440data/voting", 0, "logreg", 50)
    # boost("440data/volcanoes", 0, "nbayes", 50)
    # boost("440data/volcanoes", 0, "dtree", 50)

