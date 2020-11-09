from Tree import DecisionTreeClassifier
import numpy as np
import sys
from sklearn.datasets import load_iris

dataset = load_iris()
X, y = dataset.data, dataset.target
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X, y)