
import numpy as np
import pandas as pd
from pandas import cut,qcut

def process(X, prob, n_bin=3,cut_method='cut'):
    """
    process the data
    :param X: data X
    :param prob: problem name
    :param n_bin: number of bins
    :return:
    """
    cut_method=eval('pd.'+cut_method)
    if prob == 'spam':
        # Discretization
        for i in range(len(X[0])):
            if i != 5:
                # print(X[:,i])
                X[:, i] = cut_method(X[:, i], n_bin, retbins=True, labels=[x for x in range(n_bin)])[0].codes

        X = X.astype(np.int32)
    if prob == 'volcanoes':
        X = X.astype(np.int32)
        X=X[:,1:]#remove
        for i in range(len(X[0])):
            X[:, i] = cut_method(X[:, i], n_bin, retbins=True, labels=[x for x in range(n_bin)])[0].codes
    if prob == 'voting':
        X = X.astype(np.int32)

    return X
