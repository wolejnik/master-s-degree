import numpy as np
from TreeNode import Node
from sklearn import preprocessing
import pandas as pd
from sklearn import preprocessing
from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.tree import export_graphviz

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.criterion = 'gini'

    def _gini(self, y):
        m = y.size
        return 1.0 - np.sum(np.power((np.sum(y == c) / m), 2) for c in range(self.n_classes_))

    def _grow_tree(self, X, y, depth=0, node_id=0):
        number_samples_per_class = np.unique([np.sum(y == i) for i in range(self.n_classes_)])
        predicted_class = np.argmax(number_samples_per_class)
        print(f'predicted_class {predicted_class}')
        node = Node(gini=self._gini(y), number_samples=y.size, number_samples_per_class=number_samples_per_class,
                    predicted_class=predicted_class)

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left = X[indices_left]
                y_left = y[indices_left]
                X_right = X[~indices_left]
                y_right = y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1, node_id + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1, node_id + 1)
        return node

    def _best_split(self, X, y):
        le = preprocessing.LabelEncoder()
        m = y.size
        if m <= 1:
            return None, None
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

        # calculate ginie of the current node
        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None
        for idx in range(self.n_features_):
            tmp = le.fit(X[:, idx])
            tmp2 = le.classes_
            feature_to_binary = le.transform(X[:, idx])

            thresholds, classes = zip(*sorted(zip(feature_to_binary, y)))
            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):
                c = classes[i - 1]
                num_left[c] = num_left[c] + 1
                num_right[c] = num_right[c] - 1
                gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(self.n_classes_))
                gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_))

                # Average of children gini
                gini = (i * gini_left + (m - i) * gini_right) / m
                if thresholds[i] == thresholds[i - 1]:
                    continue
                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2  # midpoint

        return best_idx, best_thr

    def fit(self, X, y):
        self.n_classes_ = len(y)
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def _predict(self, args):
        node = self.tree_
        while node.left:
            if args[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def predict(self, X):
        return [self._predict(x) for x in X]

    def print_tree(self, feature_names, class_names, show_details=True):
        self.tree_.debug(feature_names, class_names, show_details)


if __name__ == "__main__":
    le = preprocessing.LabelEncoder()
    zoo = pd.read_csv('./zoo.csv')
    y = zoo['type'].to_numpy()
    X = zoo.drop(['animal', 'type'], axis=1).to_numpy()
    clf = DecisionTree(max_depth=50)

    tmp_y = le.fit(y)
    tmp_y2 = le.classes_
    classes_to_binary = le.transform(y)
    clf.fit(X, classes_to_binary)
    target_names = pd.unique(y)
    feature_names = zoo.drop(['type'], axis=1)
    print(feature_names.columns)
    clf.print_tree(feature_names.columns, target_names, True)
