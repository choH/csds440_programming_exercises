'''
Author: Mingyang Tie, mxt497 Shaochen (Henry) ZHONG, sxz517
Date: 2020-09-22 00:31:41
LastEditTime: 2020-09-24 21:25:34
'''

class TreeNode:
    def __init__(self, val=None, attribute=None, label=None):
        super().__init__()

        self.val = val
        self.attribute = attribute
        self.label = label
        self.attr_idx = None
        self.children = []

    def add_child(self, child):
        self.children.append(child)

    def is_leaf(self):
        return not self.children

    def set_value(self, val):
        self.val = val

    def set_attr(self, attr):
        self.attribute = attr

    def set_label(self, label):
        self.label = label

    def set_attr_idx(self, attr_idx):
        self.attr_idx = attr_idx
