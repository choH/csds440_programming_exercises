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


class ID3Decision_Tree: #Create class about ecision tree node
	def def __init__(self, parent, child, attribute, split, leaf, depth, branch):
		self.parent = parent
		self.child = child
		self.attribute = attribute
		self.split = split
		self.leaf = leaf 
		seld.depth = depth
		self.branch = branch
		
	def insert_parent(self, parent): #parents of node
		self.parent = parent #insert parent
	
	def insert_child(self, child, b):#all children in the node
		self.child.append(child) # add child in children list
		self.branch[b] = child #set values of branch of this child
		
	def insert_attribute(self, attribute):
		self.attribute = attribute #Set attribute when node split
		
	def insert_split(self, split):
		self.split = split #Set split value for attribute
		
	def insert_leaf(self, leaf):#Set class index of leaf node 
		self.leaf = leaf
		
	def insert_depth(self, depth): #depth of the node
		self.depth = depth
        
    def IG(self, split, H_split):
        pass

    def GR(self, split, y):
        H_split = entropy(split)
        IG = self.IG(split, H_split)
        GR = IG/H_split
		



