'''
Author: Mingyang Tie, mxt497 Shaochen (Henry) ZHONG, sxz517
Date: 2020-09-18 23:12:29
LastEditTime: 2020-09-25 11:18:35
'''
from tree import TreeNode
from collections import Counter
import math
from mldata import parse_c45
import numpy as np


class ID3DecisionTree:
    def __init__(self,
                 max_depth: int,
                 path: str,
                 criterion: str,
                 cv: bool = False):
        super().__init__()
        self.max_depth = max_depth
        self.criterion = criterion
        self.cv = cv

        # load data
        temp = path.split("/")
        file_base = temp[-1]

        if len(temp) == 1:
            root_dir = "."
        else:
            root_dir = "/".join(temp[:-1])
        data = parse_c45(file_base, root_dir)
        self.A = []
        self.X = []
        self.classes = []

        for index, column in enumerate(data.schema):
            if index == 0:
                continue
            if column.type == "CLASS":
                class_idx = index
            else:
                self.A.append((column.name, column.type))
        for sample in data:
            self.X.append(sample[1:class_idx] + sample[class_idx + 1:])
            self.classes.append(sample[class_idx])
        self.D = (self.X, self.classes)
        self.attr2idx = dict()
        for index, attr in enumerate(self.A):
            self.attr2idx[attr[0]] = index
        self.root = None


    @staticmethod
    def _entropy(D):
        """Get entropy of D.
        """
        _, y = D
        distribution = Counter(y)
        total = len(y)

        temp = []
        for label in distribution:
            freq = distribution[label]
            p = freq / total
            temp.append(p)
        p = np.array(temp)
        return -np.sum(p * np.log2(p))

    @staticmethod
    def _gain(D, idx, attr_type):
        """Get the information gain of idx attribute.

            idx (int): attribute index.
        """

        ent_D = ID3DecisionTree._entropy(D)

        # calculate the entropy of each attribute value
        X, y = D
        optimal_point = None

        if attr_type != "CONTINUOUS":
            buf_v = dict()
            for index, sample in enumerate(X):
                v = sample[idx]
                if v not in buf_v:
                    buf_v[v] = [[], []]
                buf_v[v][0].append(sample)
                buf_v[v][1].append(y[index])

            weighted_sum_d_v = 0.0
            total = len(y)
            for v in buf_v:
                D_v = tuple(buf_v[v])
                total_v = len(D_v[1])
                weighted_sum_d_v += total_v / total * ID3DecisionTree._entropy(
                    D_v)
        else:
            optimal_point, weighted_sum_d_v = ID3DecisionTree._get_min_entropy(
                D, idx)

        return ent_D - weighted_sum_d_v, optimal_point

    @staticmethod
    def _get_IV(D, idx, optimal_point):
        X, y = D
        total = len(y)
        if optimal_point is None:
            buf_v = Counter(map(lambda x: x[idx], X))
        else:
            buf_v = Counter(map(lambda x: int(x[idx] <= optimal_point), X))

        temp = []
        for v in buf_v:
            p = buf_v[v] / total
            temp.append(p)
        p = np.array(temp)
        return -np.sum(p * np.log2(p))

    @staticmethod
    def _gain_ratio(D, idx, attr_type):
        """Instead of using information, gain ratio is utilized.
        """
        gain, optimal_point = ID3DecisionTree._gain(D, idx, attr_type)
        IV = ID3DecisionTree._get_IV(D, idx, optimal_point)
        IV += 1e-8
        return gain / IV, optimal_point
