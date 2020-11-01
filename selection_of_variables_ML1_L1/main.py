import pandas as pd
import numpy as np
from sklearn.datasets import fetch_rcv1
from sklearn.preprocessing import Binarizer
import time

def freq(X, prob=True):
    X = np.array(X)
    xi = np.unique(X)
    ni = np.zeros_like(xi, dtype='int32')
    for i, x in enumerate(xi):
        ni[i] = np.sum(X == x)

    print(ni)
    if prob:
        ni = ni/np.sum(ni)

    return xi, ni


def freq2(X, Y, prob=True):
    X = np.array(X)
    Y = np.array(Y)
    xi = np.unique(X)
    yj = np.unique(Y)
    nij = np.zeros((xi.size, yj.size), dtype='int32')
    for i, x in enumerate(xi):
        for j, y in enumerate(yj):
            nij[i, j] = np.sum(np.logical_and(X == x, Y == y))

    if prob:
        nij = nij/np.sum(nij)

    return xi, yj, nij


def entropy(p):
    return -np.sum(p * np.log2(p+1e-16))

def infogain(p):
    return entropy(np.sum(p, axis=1)) + entropy(np.sum(p, axis=0)) - entropy(p)

def kappa(p):
    return infogain(p)/entropy(p)

def gini(p):
    return 1 - np.sum(p**2)

def ginigain(p):
    GY = gini(np.sum(p, axis=0))
    PX = p.sum(axis=1, keepdims=True)
    GYX = np.sum(PX * (1 - np.sum((p/PX)**2, axis=1)))

    return GY - GYX


if __name__ == "__main__":
    df = pd.read_csv('./zoo.csv')
    type = df['type'].to_list()

    columns_relevance = {}
    for i in range(1, 17):
        att = df.iloc[:, i].to_list()
        xi, yj, pij = freq2(att, type)
        k = kappa(pij)
        ig = infogain(pij)
        g = ginigain(pij)
        columns_relevance[df.columns[i]] = (k, g, ig)

    print(columns_relevance)
    rel = pd.DataFrame.from_dict(columns_relevance, orient='index',
                                 columns=['kappa', 'ginigain', 'infogain'])
    print(rel)

    # Zadanie 8
    rcvl = fetch_rcv1()
    X = rcvl["data"]
    Y = rcvl.target[:, 87]
    print("---")
    print(f'X:{X}, Y:{Y}')

    X = X.astype("bool")
    Y = Y.astype("bool")
    print(f'ol {X}---{Y}')

    # binarizer
    binarizer = Binarizer().fit(X)
    binarizer = Binarizer().fit(Y)
    binary_X = binarizer.transform(X)
    binary_Y = binarizer.transform(X)
    print(f'X:{binary_X}, Y:{binary_Y}')
    #
    # columns_relevance = {}
    # start = time.time()
    # for i in range(0, X.shape[1]):
    #     attributes = binary_X[:, i]
    #     # xi, yj, pij = freq2(attributes, Y)
    #     k = kappa(pij)
    #     ig = infogain(pij)
    #     g = ginigain(pij)
    #     columns_relevance[i] = (k, g, ig)
    #
    # end = time.time()
    # relations = pd.DataFrame.from_dict(columns_relevance, orient='index', columns=['kappa', 'ginigain', 'infogain'])
    # print('Whole operation for ', binary_X.shape[1], 'words took: ', end - start)
    # print(relations)