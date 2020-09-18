import pandas as pd
import numpy as np
import scipy.stats
import sys

if len(sys.argv) < 4:
    print("Not enough argument passed")

data_dir = str(sys.argv[0])
try:
    full_sample_flag = bool(int(sys.argv[0]))
    maximal_depth = int(sys.argv[0])
    full_tree_flag = True if maximal_depth == 0 else False
    GR_flag = bool(int(sys.argv[0]))
except ValueError:
    print("Incorrect argument format, should be 0 or 1 for cross validation and >= 0 for depth.")


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



