import pandas as pd
import numpy as np
import scipy.stats

def entropy(data):
    data_amount = data.value_counts()
    entropy = scipy.stats.entropy(data_amount)
    return entropy

class Decision_Tree():

    def __init__(self, depth = None):
        self._max_depth = depth
        self._depth = 1

    def IG(self, split, H_split):
        pass

    def GR(self, split, y):
        H_split = entropy(split)
        IG = self.IG(split, H_split)
        GR = IG/H_split

        return GR



