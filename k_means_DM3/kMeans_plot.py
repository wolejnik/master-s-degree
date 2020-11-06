import kMeans
import numpy as np
from random import choice
from matplotlib import pyplot
from sklearn import datasets
import scipy

def CutToTwoFeatures(items, indexA, indexB):
    n = len(items)
    X = []
    for i in range(n):
        item = items[i]
        newItem = [item[indexA], item[indexB]]
        X.append(newItem)

    return X


def PlotClusters(clusters):
    n = len(clusters)
    # Cut down the items to two dimension and store to X
    X = [[] for i in range(n)]

    for i in range(n):
        cluster = clusters[i]
        for item in cluster:
            X[i].append(item)

    colors = ['r', 'b', 'g', 'c', 'm', 'y']

    for x in X:
        # Choose color randomly from list, then remove it
        # (to avoid duplicates)
        c = choice(colors)
        colors.remove(c)

        Xa = []
        Xb = []
        for item in x:
            Xa.append(item[0])
            Xb.append(item[1])

        pyplot.plot(Xa, Xb, '.', color=c)



def plot_raw(clusters):
    n = len(clusters)
    # Cut down the items to two dimension and store to X
    X = [[] for i in range(n)]

    for i in range(n):
        cluster = clusters[i]
        for item in cluster:
            X[i].append(item)

    colors = ['r', 'b', 'g', 'c', 'm', 'y']

    for x in X:
        # Choose color randomly from list, then remove it
        # (to avoid duplicates)
        c = choice(colors)
        colors.remove(c)

        Xa = []
        Xb = []
        for item in x:
            Xa.append(item[0])
            Xb.append(item[1])

        pyplot.plot(Xa, Xb, 'o', color=c)


    def find_perm(clusters, Y_real, Y_pred):
        perm = []
        for i in range(clusters):
            idx = Y_pred == i
            new_label = scipy.stats.mode(Y_real[idx])[0][0]
            perm.append(new_label)
        return [perm[label] for label in Y_pred]

def main():
    # items = kMeans.ReadData('data.txt')
    # items = CutToTwoFeatures(items, 2, 3)

    iris = datasets.load_iris()
    X = iris.data
    Y = iris.target

    raw = []
    for i in range(0, 150, 50):
        sec = X[i:i+50]
        raw.append(sec)


    k = 3
    means = kMeans.CalculateMeans(k, X)
    clusters = kMeans.FindClusters(means, X)

    plot_raw(raw)
    PlotClusters(clusters)
    pyplot.show()

main()

