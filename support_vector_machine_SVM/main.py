import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt
import time
from sklearn.metrics import accuracy_score
import pandas as pd
from numpy import linalg as LA
import cvxopt
import warnings
warnings.filterwarnings("ignore")
from linear_classifiers import LinBase, PerceptronMargin, Perceptron, AveragedPerceptron, VotedPerceptron, plotClass


class LinSvm(LinBase):

    def __init__(self, **kwargs):
        super(LinSvm, self).__init__(**kwargs)
        self.name_ = 'LinSvm'

    def fit(self, X, y):
        m, n = X.shape
        p = np.diag(np.ones(n+1))
        p[-1, -1] = 0
        q = np.zeros(n + 1)
        g = (-1)*np.hstack((X, np.ones((m, 1))))*np.outer(y, np.ones(n + 1))
        h = (-1) * np.ones((m, 1))

        pc = cvxopt.matrix(p)
        qc = cvxopt.matrix(q)
        gc = cvxopt.matrix(g)
        hc = cvxopt.matrix(h)

        sol = cvxopt.solvers.qp(pc, qc, gc, hc)
        print(sol)

        self.coef_ = np.array(sol['x'])[0:-1, 0]
        self.intercept_ = np.array(sol['x'])[-1, 0]
        print('SVM')
        print('coef: ', self.coef_)
        print('intercept: ', self.intercept_)
        print('sol x: ', sol['x'])
        print('sol z: ', sol['z'])
        #zaznaczyc punkty pod tymi indeksami
        self.svm_inds = np.nonzero(np.array(sol['z']) > 1)[0]
        print('svm_inds')
        print(self.svm_inds)


if __name__ == "__main__":
    m, n = 200, 2
    X = np.random.rand(m, n)
    y = (X.dot([1, 1]) > 1) * 2 - 1

    for i, clf in enumerate([Perceptron(),
                             AveragedPerceptron(),
                             VotedPerceptron(),
                             PerceptronMargin(),
                             LinSvm()]):
        clf.fit(X, y)
        plotClass(X, y, clf)
        print(clf.coef_, clf.intercept_)
        a, b = clf.margin(X, y)
        print('Wartość marginesu wynosi: ', a)
    plt.show()