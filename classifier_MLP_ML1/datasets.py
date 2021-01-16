import numpy as np
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from random import random
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_circles as make_circle
from time import time


def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.2, test_size=0.8)
    return x_train, x_test, y_train, y_test


def make_checkerboard(p, dim, noise=0):
    x = np.random.rand(p, 2) * dim
    y = np.array([0 if (int(item[0])+int(item[1])) % 2 == 0 else 1 for item in x])
    x = x + np.random.randn(*x.shape) * noise
    return x, y


def make_spiral(p, noise=1):
    t = np.linspace(0, 2*2*np.pi, p)
    x1 = np.array([t*np.cos(t), t*np.sin(t)]).T
    x2 = np.array([t*np.cos(t+np.pi), t*np.sin(t+np.pi)]).T
    x = np.vstack((x1, x2))
    y = np.concatenate((np.zeros(p), np.ones(p)))
    x = x + np.random.randn(*x.shape) * noise
    return x, y


def plot_decision_boundry(data, classifier):
    x_min, x_max = data[:, 0].min() - 0.1, data[:, 0].max() + 0.1
    y_min, y_max = data[:, 1].min() - 0.1, data[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                         np.arange(y_min, y_max, .02))
    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)


def plot_class(X, y, clf):
    colors = ["red" if item == True else "blue" for item in y]
    plt.figure()
    plot_decision_boundry(x_train, clf)
    plt.scatter(X[:, 0], X[:, 1], c=colors)
    plt.show()

if __name__ == '__main__':
    # x, y = make_spiral(500)
    x, y = make_checkerboard(500, 3)

    x_train, x_test, y_train, y_test = split_data(x, y)
    clf = MLPClassifier(hidden_layer_sizes=(100, 50, 20), solver='adam', activation='relu', alpha=1e-5, max_iter=1000)
    clf.fit(x_train, y_train)
    plot_class(x_test, y_test, clf)
