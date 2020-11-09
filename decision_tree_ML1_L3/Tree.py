import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
import TreeNode
from sklearn import tree
import pandas as pd
from sklearn import preprocessing

class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        """Build decision tree classifier."""
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._print_tree(X, y)


    def debug(self, feature_names, class_names):
        self.tree_.debug(feature_names, class_names)

    def _gini(self, y):
        m = y.size
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in range(self.n_classes_))

    def _best_split(self, X, y):
        """
        Find the best split for a node.
        Returns:
            best_idx: Index of the feature for best split, or None if no split is found.
            best_thr: Threshold to use for the split, or None if no split is found.
        """
        # Need at least two elements to split a node.
        if y.size <= 1:
            return None, None

        # Count of each class in the current node.
        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

        # Gini of current node.
        best_gini = 1.0 - sum((n / y.size) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        # Loop through all features.
        for idx in range(self.n_features_):
            # Sort data along selected feature.
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, y.size):  # possible split positions
                print(f'classes[i - 1] ->>> {classes[i - 1]}')
                c = classes[i - 1]
                num_left[classes.index(c)] += 1
                num_right[classes.index(c)] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (y.size - i)) ** 2 for x in range(self.n_classes_)
                )

                # The Gini impurity of a split is the weighted average of the Gini impurity of the children.
                gini = (i * gini_left + (y.size - i) * gini_right) / y.size

                # The following condition is to make sure we don't try to split two points with identical values for that feature
                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2

        return best_idx, best_thr

    def _print_tree(self, X, y, depth=0):
        """Build a decision tree by recursively finding the best split."""
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = TreeNode.Node(
            gini=self._gini(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        # Split recursively until maximum depth is reached.
        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._print_tree(X_left, y_left, depth + 1)
                node.right = self._print_tree(X_right, y_right, depth + 1)
        return node

    # def predict(self, X):
    #     """Predict class for X."""
    #     return [self._predict(inputs) for inputs in X]
    #
    # def _predict(self, inputs):
    #     """Predict class for a single sample."""
    #     node = self.tree_
    #     while node.left:
    #         if inputs[node.feature_index] < node.threshold:
    #             node = node.left
    #         else:
    #             node = node.right
    #     return node.predicted_class


if __name__ == "__main__":

    # dataset = load_iris()
    # X, y = dataset.data, dataset.target
    # clf = DecisionTreeClassifier(max_depth=2)
    # clf.fit(X, y)

    # Zoo

    df = pd.read_csv('./zoo.csv')
    animal_type = df['type'].to_list()
    np.unique(animal_type)
    X_df = df.drop(columns=['legs', 'animal', 'type'])
    X = X_df.to_numpy()
    y = df['type'].to_numpy()
    # y = np.where(y=='insect', 1, 0)  # jeżeli chcemy wykrywać tylko binarnie czy dany typ zwierzęcia czy nie
    print(y)

    # le = preprocessing.LabelEncoder()
    # le.fit(y)
    # y = le.transform(y)

    clf = DecisionTreeClassifier(max_depth=2)
    clf.fit(X, y)

    # 3. Predict.
    # pred = clf.predict([input])[0]
    # print("Input: {}".format(input))
    # print("Prediction: " + dataset.target_names[pred])


    clf.debug(
        list(X),
        list(y)
    )

    # tree.export_graphviz(clf,
    #                      out_file="tree.dot",
    #                      feature_names=dataset.feature_names,
    #                      class_names=dataset.target_names)