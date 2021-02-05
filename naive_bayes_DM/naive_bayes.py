import numpy as np
import numpy
import sys

from sklearn.metrics import accuracy_score

numpy.set_printoptions(threshold=sys.maxsize)
from loading_data import load_dataset

class NaiveBayes:
    def __init__(self, X, y):
        self.num_emails, self.num_words = X.shape
        self.classes = np.unique(y_train)
        self.eps = 1e-6

    def fit(self, X, y):
        self.classes_mean = {}
        self.classes_likrlihood = {}
        self.classes_prior = {}

        for c in self.classes:
            X_c = X[y == c]
            self.classes_mean[str(c)] = np.mean(X_c, axis=0)
            self.classes_likrlihood[str(c)] = np.var(X_c, axis=0)
            self.classes_prior[str(c)] = X_c.shape[0] / X.shape[0]

    def predict(self, X):
        probs = np.zeros((self.num_emails, self.classes))

        for c in range(self.classes):
            prior = self.classes_prior[str(c)]
            print(f'prior => {prior}')

            probs_c = self.density_function(
                X, self.classes_mean[str(c)], self.classes_likrlihood[str(c)]
            )
            print(f'probs_c => {probs_c}')

            probs[:, c] = probs_c + np.log(prior)

        return np.argmax(probs, 1)

    def density_function(self, x, mean, sigma):
        # Calculate probability from Gaussian density function
        const = -self.num_words / 2 * np.log(2 * np.pi) - 0.5 * np.sum(
            np.log(sigma + self.eps)
        )
        probs = 0.5 * np.sum(np.power(x - mean, 2) / (sigma + self.eps), 1)
        return const - probs

if __name__ == "__main__":
    # For spam emails (Make sure to run build_vocab etc. to have .npy files)
    # X = np.load("X.npy")
    # y = np.load("y.npy")

    subsets = ["", "50", "100", "400"]
    for subset in subsets:
        X_train, y_train, X_test, y_test = load_dataset(train_subset=subset)
        NB = NaiveBayes(X_train, y_train)
        NB.fit(X_test, y_train)
        y_pred = NB.predict(X_test)

        print(f"Acc {subset}: {accuracy_score(y_test, y_pred)}")
    #
    # print(f'Number of spam messages => {sum(y)}, {sum(y)/y.shape[0]*100}%')
    # print(f"Accuracy: {sum(y_pred==y)/X.shape[0]}")