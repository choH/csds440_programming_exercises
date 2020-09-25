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

    def train(self, D):
        self.root = self._TreeGenerate(D, self.A)

    def _test_helper(self, root, x):
        if not root.children:
            return root.label

        attr_idx = root.attr_idx
        for child in root.children:
            if isinstance(child.val, tuple):
                if child.val[0] == "<" and x[attr_idx] <= child.val[1]:
                    return self._test_helper(child, x)
            else:
                if x[attr_idx] == child.val:
                    return self._test_helper(child, x)

        return self.classes[0]

    def test(self, X_test):
        y_test = []

        for index, x in enumerate(X_test):
            y_test.append(self._test_helper(self.root, x))

        return y_test

    def run(self):
        if not self.cv:
            self.train(self.D)
            y_test = self.test(self.X)
            accuracy = ID3DecisionTree._get_accuracy(y_test, self.classes)
            size, depth, first_feature = self._get_tree_info()
            ID3DecisionTree._display(accuracy, size, depth, first_feature)
        else:
            # cross validation
            FOLD = 5
            folds = self._split_data(FOLD)
            accs = []

            for k in range(FOLD):
                D_train = [[], []]
                D_test = [[], []]

                for i in range(FOLD):
                    fold = folds[i]
                    if i == k:
                        D_test = fold
                    else:
                        D_train[0] += fold[0]
                        D_train[1] += fold[1]

                D_train = tuple(D_train)
                D_test = tuple(D_test)

                self.train(D_train)
                X_test, y_label = D_test
                y_test = self.test(X_test)
                accuracy = ID3DecisionTree._get_accuracy(y_test, y_label)
                accs.append(accuracy)
                size, depth, first_feature = self._get_tree_info()
                print("====================================")
                ID3DecisionTree._display(accuracy, size, depth, first_feature)
            print("====================================")
            print("Average Accuracy: %.3f" % (sum(accs) / FOLD))

    def _split_data(self, N):
        # split into two parts
        negative, positive = [], []
        labels = sorted(set(self.classes))
        for index, label in enumerate(self.classes):
            if label == labels[0]:
                negative.append(self.X[index])
            else:
                positive.append(self.X[index])

        # N folds
        folds = []
        pos_total = len(positive)
        neg_total = len(negative)
        pos_step = pos_total // N
        neg_step = neg_total // N
        pos_start = 0
        neg_start = 0
        while neg_start < neg_total and pos_start < pos_total:
            pos_end = min(pos_total, pos_start + pos_step)
            neg_end = min(neg_total, neg_start + neg_step)
            fold = [[], []]
            fold[0] += positive[pos_start:pos_end]
            fold[1] += [labels[0]] * (pos_end - pos_start)
            fold[0] += negative[neg_start:neg_end]
            fold[1] += [labels[1]] * (neg_end - neg_start)
            pos_start += pos_step
            neg_start += neg_step
            folds.append(fold)
        return folds

    @staticmethod
    def _display(accuracy, size, depth, first_feature):
        print("Accuracy: %.3f" % accuracy)
        print(f"Size: {size}")
        print(f"Maximum Depth: {depth}")
        print(f"First Feature: <{first_feature}>")

    def _get_tree_info(self):
        size = ID3DecisionTree._get_size(self.root)
        depth = ID3DecisionTree._get_max_depth(self.root)
        first_feature = self.root.attribute

        return size, depth, first_feature

    @staticmethod
    def _get_max_depth(root):
        if not root.children:
            return 0

        depth = 0
        for child in root.children:
            depth = max(depth, 1 + ID3DecisionTree._get_max_depth(child))

        return depth

    @staticmethod
    def _get_size(root):
        if not root.children:
            return 1
        size = 1
        for child in root.children:
            size += ID3DecisionTree._get_size(child)
        return size

    @staticmethod
    def _get_accuracy(preds, labels):
        correct = 0
        total = len(preds)

        for index, pred in enumerate(preds):
            if pred == labels[index]:
                correct += 1

        return correct / total

    def _TreeGenerate(self, D, A, v=None, depth=0):
        """Generate a decision tree. Note that all data vectors are Lists.

        Args:
            D (Tuple[x, y]): x is feature vectors and y is label vectors.
            A (List[Tuple(name, type)]): arrtributes index list.
        """
        root = TreeNode()
        root.set_value(v)

        X, y = D

        # all the samples belong to the same class
        if len(set(y)) == 1:
            root.set_label(y[0])
            return root

        # A is empty set or all the samples are the same on arrtributes set A or achieving the maximum depth
        if len(A) == 0 or ID3DecisionTree._is_same(X) or (
                self.max_depth > 0 and depth + 1 > self.max_depth):
            most_common_class = Counter(y).most_common(1)[0][0]
            root.set_label(most_common_class)
            return root

        # get the optimal spliting attribute
        optimal_attribute_idx, optimal_point = self._get_optimal_attribute_idx(
            D, A)

        buf_v = dict()
        for index, sample in enumerate(X):
            v = sample[optimal_attribute_idx]

            if A[optimal_attribute_idx][1] != "CONTINUOUS":
                if v not in buf_v:
                    buf_v[v] = [[], []]
                buf_v[v][0].append(sample[:optimal_attribute_idx] +
                                   sample[optimal_attribute_idx + 1:])
                buf_v[v][1].append(y[index])
            else:
                if "pos" not in buf_v:
                    buf_v["pos"] = [[], []]
                    buf_v["neg"] = [[], []]

                if v <= optimal_point:
                    buf_v["neg"][0].append(sample[:optimal_attribute_idx] +
                                           sample[optimal_attribute_idx + 1:])
                    buf_v["neg"][1].append(y[index])
                else:
                    buf_v["pos"][0].append(sample[:optimal_attribute_idx] +
                                           sample[optimal_attribute_idx + 1:])
                    buf_v["pos"][1].append(y[index])

        # set the optimal attribute for root
        root.set_attr(A[optimal_attribute_idx][0])
        root.set_attr_idx(self.attr2idx[A[optimal_attribute_idx][0]])
        new_A = A[:optimal_attribute_idx] + A[optimal_attribute_idx + 1:]

        for v in buf_v:
            D_v = tuple(buf_v[v])
            if A[optimal_attribute_idx][1] != "CONTINUOUS":
                if D_v[0]:
                    root.add_child(self._TreeGenerate(D_v, new_A, v,
                                                      depth + 1))
            else:
                if D_v[0]:
                    if v == "pos":
                        root.add_child(
                            self._TreeGenerate(D_v, new_A,
                                               (">", optimal_point),
                                               depth + 1))
                    else:
                        root.add_child(
                            self._TreeGenerate(D_v, new_A,
                                               ("<", optimal_point),
                                               depth + 1))

        return root

    def _get_optimal_attribute_idx(self, D, A):
        """Get the optimal attribute.
        """
        attributes_num = len(D[0][0])

        if self.criterion == "gain":
            criterion_func = ID3DecisionTree._gain
        elif self.criterion == "gain ratio":
            criterion_func = ID3DecisionTree._gain_ratio
        else:
            criterion_func = ID3DecisionTree._gini

        maxm_criterion = float('-inf')
        optimal_attribute_idx = -1
        optimal_point = None

        for idx in range(attributes_num):
            cur_criterion, cur_optimal_point = criterion_func(
                D, idx, A[idx][1])
            if cur_criterion > maxm_criterion:
                maxm_criterion = cur_criterion
                optimal_attribute_idx = idx
                optimal_point = cur_optimal_point

        return optimal_attribute_idx, optimal_point

    @staticmethod
    def _is_same(X):
        X_ = map(lambda x: tuple(x), X)
        return len(set(X_)) == 1

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
    def _get_min_entropy(D, idx):
        """Get the split point with the minimum entropy.
        """
        X, y = D
        total = len(y)

        # get sorted value for idx attribute
        sorted_attributes = sorted(map(lambda x: x[idx], X))

        # get candidate split points set
        candidates = set()

        for index in range(len(sorted_attributes) - 1):
            candidates.add(
                (sorted_attributes[index] + sorted_attributes[index + 1]) / 2)

        # sort samples
        samples = []
        for index, x in enumerate(X):
            samples.append((x, y[index], x[idx]))
        samples = sorted(samples, key=lambda x: x[2])

        # get the optimal split point
        optimal_point = None
        minm_entropy = float('inf')
        buf_v = dict()
        buf_v["neg"] = [[], []]
        buf_v["pos"] = [[], []]
        for index, sample in enumerate(samples):
            buf_v["pos"][0].append(sample[0])
            buf_v["pos"][1].append(sample[1])
        for index, point in enumerate(sorted(set(candidates))):
            cur_entropy = 0.0

            # pop samples from positive to negative for equal condition
            while buf_v["pos"][0] and (
                    abs(buf_v["pos"][0][0][idx] - point) < 1e-7
                    or buf_v["pos"][0][0][idx] <= point):
                buf_v["neg"][0].append(buf_v["pos"][0][0])
                buf_v["neg"][1].append(buf_v["pos"][1][0])
                del buf_v["pos"][0][0]
                del buf_v["pos"][1][0]

            for v in buf_v:
                D_v = tuple(buf_v[v])
                total_v = len(D_v[1])
                cur_entropy += total_v / total * ID3DecisionTree._entropy(D_v)

            if cur_entropy < minm_entropy:
                minm_entropy = cur_entropy
                optimal_point = point
        return optimal_point, minm_entropy

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
    def _gini(D, idx, attr_type):
        # calculate the entropy of each attribute value
        X, y = D
        optimal_point = None
        total = len(X)
        gini = 1

        if attr_type != "CONTINUOUS":
            count = Counter()
            for sample in X:
                count[sample[idx]] += 1

            for freq in count:
                gini -= (count[freq] / total)**2
        else:
            optimal_point, weighted_sum_d_v = ID3DecisionTree._get_max_gini(
                D, idx)

        return gini, optimal_point

    @staticmethod
    def _get_max_gini(D, idx):
        X, y = D
        total = len(y)

        # get sorted value for idx attribute
        sorted_attributes = sorted(map(lambda x: x[idx], X))

        # get candidate split points set
        candidates = set()

        for index in range(len(sorted_attributes) - 1):
            candidates.add(
                (sorted_attributes[index] + sorted_attributes[index + 1]) / 2)

        # sort samples
        samples = []
        for index, x in enumerate(X):
            samples.append((x, y[index], x[idx]))
        samples = sorted(samples, key=lambda x: x[2])

        # get the optimal split point
        optimal_point = None
        max_gini = float("-inf")
        buf_v = []
        for index, sample in enumerate(samples):
            buf_v.append(sample[0])
        negative = 0
        for index, point in enumerate(sorted(set(candidates))):

            # pop samples from positive to negative for equal condition
            while buf_v and (abs(buf_v[0][idx] - point) < 1e-7
                             or buf_v[0][idx] <= point):
                del buf_v[0]
                negative += 1

            positive = total - negative
            cur_gini = 1 - (negative / total)**2 - (positive / total)**2

            if cur_gini > max_gini:
                max_gini = cur_gini
                optimal_point = point
        return optimal_point, max_gini

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
        if IV == 0:
            IV = float("inf")
        return gain / IV, optimal_point
