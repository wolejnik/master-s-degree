from Tree import Tree
import numpy as np
import sys
from sklearn.datasets import load_iris

dataset = load_iris()
X, y = dataset.data, dataset.target
clf = Tree(max_depth=5)
clf.fit(X, y)
print(clf.predict([[0, 0, 5, 1.5]]))