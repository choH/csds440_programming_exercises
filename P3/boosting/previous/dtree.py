
import sys
from ID3_dtree import ID3DecisionTree
import random

random.seed(12345)

# parse args
if len(sys.argv) != 5:
    raise ValueError("Not proper command!")
_, path, cv, max_depth, criterion = sys.argv

try:
    cv = int(cv)
    if cv:
        cv = False
    else:
        cv = True

    max_depth = int(max_depth)
    if max_depth < 0:
        raise ValueError("Not valid maximum depth.")
    criterion = int(criterion)

    if criterion:
        criterion = "gain ratio"
    else:
        criterion = "gain"

except ValueError:
    print("Invalid arguments.")

# initialize a model instance
ID3Tree = ID3DecisionTree(max_depth, path, criterion, cv)

# run the model
ID3Tree.run()
