import pandas as pd
import numpy as np
import scipy.stats

def entropy(data):
    data_amount = data.value_counts()
    entropy = scipy.stats.entropy(data_amount)
    return entropy

