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
