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


# numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html?fbclid=
#

class LinBase(BaseEstimator, ClassifierMixin):
    def __init__(self, coef=None, intercept=None, class_labels=None):
        self.coef_ = coef
        self.intercept_ = intercept
        self.class_labels_ = class_labels

    def margin(self, X, y):
        m = y * (X.dot(self.coef_) + self.intercept_)
        m = m / LA.norm(self.coef_)
        return np.min(m), m

    def decision_function(self, X):
        a, b = self.margin(X, np.ones(X.shape[0]))
        return b

    def predict_proba(self, X):
        a, b = self.margin(X, np.ones((X.shape[0],)))
        return np.array([1 - 1 / (1 + np.exp(-b)), 1 / (1 + np.exp(-b))]).T

    def predict(self, X):
        results = np.sign(X.dot(self.coef_) + self.intercept_)
        results_mapped = self.class_labels_[1 * (results > 0)]
        return results_mapped

    def get_params(self, deep=True):
        pass

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __str__(self):
        return "{} [ w={}, b={} ] ".format(self.__class__.__name__, self.coef_, self.intercept_)


class PerceptronMargin(LinBase):
    def __init__(self, break_time=10, gamma=0.5, **kwargs):
        super(PerceptronMargin, self).__init__(**kwargs)
        self.break_time_ = break_time
        self.gamma_ = gamma
        self.name_ = 'PerceptronMargin'

    def fit(self, X, y):
        m, n = X.shape
        self.class_labels_ = np.unique(y)
        if self.coef_ is not None:
            w = self.coef_
        else:
            w = np.zeros(n)
            self.coef_ = w
        if self.intercept_ is not None:
            b = self.intercept_
        else:
            b = 0
            self.intercept_ = b
        i = 0
        k = 0
        t1 = time.time()
        counter_weight, counter_iter = 0, 0

        while k < m:
            counter_iter += 1
            if y[i] * (w.dot(X[i, :]) + b) <= self.gamma_ * LA.norm(w):
                counter_weight += 1
                w = w + X[i, :] * y[i]
                b = b + y[i]
                k = 0
            else:
                k += 1
            i = (i + 1) % m
            t2 = time.time()
            if t2 - t1 > self.break_time_:
                break
        print('Perceptron z marginesem')
        print('Time: ' + str(t2 - t1))
        print('Iterations: ', counter_iter)
        print('Updates: ', k)
        print('Changes of weights: ', counter_weight)

        self.coef_ = w
        self.intercept_ = b


class Perceptron(LinBase):
    def __init__(self, break_time=10, **kwargs):
        super(Perceptron, self).__init__(**kwargs)
        self.break_time_ = break_time
        self.name_ = 'Perceptron'

    def fit(self, X, y):
        m, n = X.shape
        self.class_labels_ = np.unique(y)
        if self.coef_ is not None:
            w = self.coef_
        else:
            w = np.zeros(n)
        if self.intercept_ is not None:
            b = self.intercept_
        else:
            b = 0
        i = 0
        k = 0
        t1 = time.time()
        counter_weight, counter_iter = 0, 0

        while k < m:
            counter_iter += 1
            if (y[i] * (w.dot(X[i, :]) + b)) <= 0:
                counter_weight += 1
                w = w + X[i, :] * y[i]
                b = b + y[i]
                k = 0
            else:
                k += 1
            i = (i + 1) % m
            t2 = time.time()
            if t2 - t1 > self.break_time_:
                print('Uwaga, przekroczono czas')
                break
        print('Perceptron')
        print('Time: ' + str(t2 - t1))
        print('Iterations: ', counter_iter)
        print('Updates: ', k)
        print('Changes of weights: ', counter_weight)

        self.coef_ = w
        self.intercept_ = b


class AveragedPerceptron(LinBase):
    def __init__(self, break_time=10, k_max=1000, **kwargs):
        super(AveragedPerceptron, self).__init__(**kwargs)
        self.break_time_ = break_time
        self.v_ = None
        self.v_bias_ = None
        self.k_max = k_max
        self.name_ = 'AveragedPerceptron'

    def fit(self, X, y):
        m, n = X.shape
        self.class_labels_ = np.unique(y)
        if self.coef_ is not None:
            w = self.coef_
        else:
            w = np.zeros(n)
        if self.intercept_ is not None:
            b = self.intercept_
        else:
            b = 0
        self.v_ = 0 * w
        self.v_bias_ = 0 * b
        i = 0
        k = 0
        counter_weight, counter_iter = 0, 0
        t1 = time.time()

        while k < self.k_max:
            counter_iter += 1
            if (y[i] * (w.dot(X[i, :]) + b)) <= 0:
                self.v_ += counter_weight * w
                self.v_bias_ += counter_weight * b
                w = w + X[i, :] * y[i]
                b = b + y[i]
                k += 1
                counter_weight = 1
            else:
                counter_weight += 1
            i = (i + 1) % m
            t2 = time.time()
            if t2 - t1 > self.break_time_:
                print('Uwaga, przekroczono czas')
                break

        self.v_ += counter_weight * w
        self.v_bias_ += counter_weight * b

        print('Averaged perceptron')
        print('Time: ' + str(t2 - t1))
        print('Iterations: ', counter_iter)
        print('Updates: ', k)
        print('Changes of weights: ', counter_weight)

        self.coef_ = self.v_
        self.intercept_ = self.v_bias_


class VotedPerceptron(LinBase):
    def __init__(self, break_time=10, k_max=1000, **kwargs):
        super(VotedPerceptron, self).__init__(**kwargs)
        self.break_time_ = break_time
        self.v_ = None
        self.v_bias_ = None
        self.k_max = k_max
        self.counter_weights = np.zeros(self.k_max + 1)
        self.name_ = 'VotedPerceptron'

    def fit(self, X, y):
        m, n = X.shape
        self.v_ = np.zeros((self.k_max + 1, n))
        self.v_bias_ = np.zeros(self.k_max + 1)
        self.class_labels_ = np.unique(y)
        if self.coef_ is not None:
            w = self.coef_
        else:
            w = np.zeros(n)
        if self.intercept_ is not None:
            b = self.intercept_
        else:
            b = 0
        i = 0
        k = 0
        counter_weight, counter_iter = 0, 0
        t1 = time.time()

        while k < self.k_max:
            counter_iter += 1
            if (y[i] * (w.dot(X[i, :]) + b)) <= 0:
                self.v_[k, :] += w
                self.v_bias_[k] += b
                self.counter_weights[k] = counter_weight
                # self.v_ += counter_weight * w
                # self.v_bias_ += counter_weight * b
                w = w + X[i, :] * y[i]
                b = b + y[i]
                k += 1
                counter_weight = 1
            else:
                counter_weight += 1
            i = (i + 1) % m
            t2 = time.time()
            if t2 - t1 > self.break_time_:
                print('Uwaga, przekroczono czas')
                break

        self.v_[k, :] += w
        self.v_bias_[k] += b
        self.counter_weights[k] = counter_weight

        print('Voted Perceptron')
        print('Time: ' + str(t2 - t1))
        print('Iterations: ', counter_iter)
        print('Updates: ', k)
        print('Changes of weights: ', counter_weight)

        self.coef_ = self.v_
        self.intercept_ = self.v_bias_

    def predict(self, X):
        v = np.hstack((self.v_, np.array([self.v_bias_]).T))
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        return np.sign(np.dot(np.sign(np.dot(v, X.T)).T, self.counter_weights))

    def predict_proba(self, X):
        v = np.hstack((self.v_, np.array([self.v_bias_]).T))
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        a = np.sign(np.dot(np.sign(np.dot(v, X.T)).T, self.counter_weights))
        b = 1 - a
        return np.vstack((a, b)).T

    def margin(self, X, y):
        return 0, 0


def plotClass(X, y, clf=None, marker=["ro", "k+"], n=50):
    plt.figure()
    uy = np.unique(y)
    for i, u in enumerate(uy):
        plt.plot(X[y == u, 0], X[y == u, 1], marker[i], label=str(u))

    if clf is not None:
        if clf.name_ == 'LinSvm':
            plt.scatter(X[clf.svm_inds, 0], X[clf.svm_inds, 1], s=100,
                        facecolors='none', zorder=10, edgecolors='k')

        # if (isinstance(clf, LinSvm)):
        # plt.plot(self.svm_inds, '*')
        # pass
        # napisac plota dla punktow clf.svm+inds *
        X1, X2 = np.meshgrid(np.linspace(np.min(X[:, 0]), np.max(X[:, 0]), n),
                             np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), n))
        XX = np.array([X1.ravel(), X2.ravel()]).T
        yy = clf.predict_proba(XX)[:, 1]
        yy = np.reshape(yy, X1.shape)
        plt.contour(X1, X2, yy, [0.5])

    else:
        pass

    # plt.show()

    class LinSvm(LinBase):

        def __init__(self, **kwargs):
            super(LinSvm, self).__init__(**kwargs)
            self.name_ = 'LinSvm'

        def fit(self, X, y):
            m, n = X.shape
            p = np.diag(np.ones(n + 1))
            p[-1, -1] = 0
            q = np.zeros(n + 1)
            g = (-1) * np.hstack((X, np.ones((m, 1)))) * np.outer(y, np.ones(n + 1))
            h = (-1) * np.ones((m, 1))

            pc = cvxopt.matrix(p)
            qc = cvxopt.matrix(q)
            gc = cvxopt.matrix(g)
            hc = cvxopt.matrix(h)

            sol = cvxopt.solvers.qp(pc, qc, gc, hc)
            print(sol)

            self.coef_ = np.array(sol['x'])[0:-1, 0]
            self.intercept_ = np.array(sol['x'])[-1, 0]
            print('Svm')
            print('coef: ', self.coef_)
            print('intercept: ', self.intercept_)
            print('sol x: ', sol['x'])
            print('sol z: ', sol['z'])
            # zaznaczyc punkty pod tymi indeksami
            self.svm_inds = np.nonzero(np.array(sol['z']) > 1)[0]
            print('svm_inds')
            print(self.svm_inds)


if __name__ == "__main__":
    m, n = 200, 2
    # X = np.random.rand(m, n)
    # y = (X.dot([1, 1]) > 1) * 2 - 1
    X1 = np.random.rand(m, n)
    X2 = np.random.rand(m, n)
    X2[:, 1] = X2[:, 1] + 1.2
    X = np.vstack((X1, X2))
    y = np.concatenate((-np.ones(m), np.ones(m)))

    for i, clf in enumerate([Perceptron(),
                             AveragedPerceptron(),
                             VotedPerceptron(),
                             PerceptronMargin()]):
        clf.fit(X, y)
        plotClass(X, y, clf)
        print(clf.coef_, clf.intercept_)
        a, b = clf.margin(X, y)
        print('Wartość marginesu wynosi: ', a)

    print('SONAR')
    sonar = pd.read_csv("./sonar.csv")
    X = np.array(sonar.iloc[:, 0:-1])
    y = np.array(sonar.iloc[:, -1])

    y[y == 'Rock'] = -1
    y[y == 'Mine'] = 1

    for i, clf in enumerate([Perceptron(),
                             AveragedPerceptron(),
                             VotedPerceptron()]):
        clf.fit(X, y)

        y_pred = clf.predict(X)
        print('Predicted y')
        print(y_pred)
        print('Dokładność klasyfikatora wynosi: ', accuracy_score(list(y), list(y_pred)))

        print('clf.coef_ i clf.intercept_')
        print(clf.coef_, clf.intercept_)
        # a, b = clf.margin(X,y)

        a, b = clf.margin(X, y)
        print('Wartość marginesu wynosi: ', a)

